"""
Transfer Learning for Battery SOH GPR.

Strategy:
  1. Train GPR on source batteries (B0005, B0006, B0007) combined
     → extract optimised kernel hyperparameters (length_scale, noise_level)
  2. Initialise a new GPR for B0018 with those hyperparameters FIXED
     (no re-optimisation), fitted on only the first N cycles of B0018
  3. Compare against a "scratch" GPR trained on the same N cycles
     with default hyperparameters and full optimisation
  4. Evaluate both on the remaining B0018 cycles
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "saved")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_soh(battery_ids):
    """Return (X, y) arrays for the given batteries from saved prediction CSVs."""
    Xs, ys = [], []
    for bid in battery_ids:
        path = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
        df = pd.read_csv(path).sort_values("cycle")
        Xs.append(df[["cycle"]].values.astype(float))
        ys.append(df["soh"].values.astype(float))
    return np.vstack(Xs), np.concatenate(ys)


def _mape_per_cycle(df_eval, y_pred):
    """Return a Series of cumulative-window MAPE up to each cycle."""
    actual = df_eval["soh"].values
    mape = np.abs((actual - y_pred) / actual) * 100
    return pd.Series(mape, index=df_eval["cycle"].values)


# ── source training ───────────────────────────────────────────────────────────

def train_source_gpr(source_ids=("B0005", "B0006", "B0007")):
    """
    Train a single GPR on all source-battery cycles combined.
    Returns (fitted_gpr, scaler, kernel_params dict).
    """
    X, y = _load_soh(source_ids)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kernel = Matern(length_scale=50.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=1e-6,
    )
    gpr.fit(Xs, y)

    # Extract optimised hyperparameters
    k = gpr.kernel_
    ls   = float(k.k1.length_scale)
    nl   = float(k.k2.noise_level)
    params = {"length_scale": ls, "noise_level": nl}
    print(f"  Source kernel → length_scale={ls:.4f}  noise_level={nl:.6f}")

    joblib.dump({"model": gpr, "scaler": scaler, "params": params},
                os.path.join(MODELS_DIR, "source_gpr.pkl"))
    return gpr, scaler, params


# ── transfer-learning fit ─────────────────────────────────────────────────────

def fit_transfer_gpr(target_id="B0018", n_seed=10, source_params=None):
    """
    Fit a GPR on the first `n_seed` cycles of `target_id` using
    kernel hyperparameters transferred from source batteries.

    Returns a dict with:
      transfer_df  – predictions over all target cycles
      scratch_df   – same but for a from-scratch GPR
      mape_transfer, mape_scratch – per-cycle MAPE Series on held-out cycles
      crossover_cycle – first cycle where transfer MAPE < 2 % (or None)
    """
    path = os.path.join(RESULTS_DIR, f"{target_id}_predictions.csv")
    df = pd.read_csv(path).sort_values("cycle").reset_index(drop=True)

    seed   = df.iloc[:n_seed]
    eval_  = df.iloc[n_seed:]

    X_seed = seed[["cycle"]].values.astype(float)
    y_seed = seed["soh"].values.astype(float)
    X_eval = eval_[["cycle"]].values.astype(float)
    X_all  = df[["cycle"]].values.astype(float)

    # ── Transfer GPR ──────────────────────────────────────────────────────────
    if source_params is None:
        src_path = os.path.join(MODELS_DIR, "source_gpr.pkl")
        if os.path.exists(src_path):
            source_params = joblib.load(src_path)["params"]
        else:
            raise FileNotFoundError("Source model not found. Call train_source_gpr() first.")

    ls = source_params["length_scale"]
    nl = source_params["noise_level"]

    scaler_t = StandardScaler()
    X_seed_s = scaler_t.fit_transform(X_seed)
    X_eval_s = scaler_t.transform(X_eval)
    X_all_s  = scaler_t.transform(X_all)

    # Fix hyperparameters: skip optimizer entirely so scipy result object isn't touched
    transfer_kernel = (
        Matern(length_scale=ls, length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(noise_level=nl, noise_level_bounds="fixed")
    )
    gpr_t = GaussianProcessRegressor(
        kernel=transfer_kernel,
        optimizer=None,
        normalize_y=True,
        alpha=1e-6,
    )
    gpr_t.fit(X_seed_s, y_seed)

    y_pred_t_all, y_std_t_all = gpr_t.predict(X_all_s, return_std=True)
    y_pred_t_eval = gpr_t.predict(X_eval_s)

    # ── Scratch GPR ───────────────────────────────────────────────────────────
    scaler_s = StandardScaler()
    X_seed_ss = scaler_s.fit_transform(X_seed)
    X_eval_ss = scaler_s.transform(X_eval)
    X_all_ss  = scaler_s.transform(X_all)

    scratch_kernel = Matern(length_scale=50.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
    gpr_s = GaussianProcessRegressor(
        kernel=scratch_kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=1e-6,
    )
    gpr_s.fit(X_seed_ss, y_seed)

    y_pred_s_all, y_std_s_all = gpr_s.predict(X_all_ss, return_std=True)
    y_pred_s_eval = gpr_s.predict(X_eval_ss)

    # ── Build result DataFrames ───────────────────────────────────────────────
    transfer_df = df[["cycle", "soh"]].copy()
    transfer_df["soh_pred"]   = y_pred_t_all
    transfer_df["soh_std"]    = y_std_t_all
    transfer_df["region"]     = "eval"
    transfer_df.loc[:n_seed - 1, "region"] = "seed"

    scratch_df = df[["cycle", "soh"]].copy()
    scratch_df["soh_pred"]  = y_pred_s_all
    scratch_df["soh_std"]   = y_std_s_all
    scratch_df["region"]    = "eval"
    scratch_df.loc[:n_seed - 1, "region"] = "seed"

    # ── Per-cycle MAPE on eval set ────────────────────────────────────────────
    mape_t = _mape_per_cycle(eval_, y_pred_t_eval)
    mape_s = _mape_per_cycle(eval_, y_pred_s_eval)

    # Crossover: first eval cycle where transfer MAPE drops below 2 %
    below2 = mape_t[mape_t < 2.0]
    crossover_cycle = int(below2.index[0]) if len(below2) > 0 else None

    mae_t = mean_absolute_error(eval_["soh"], y_pred_t_eval)
    mae_s = mean_absolute_error(eval_["soh"], y_pred_s_eval)
    print(f"  Transfer MAE={mae_t:.5f}   Scratch MAE={mae_s:.5f}")
    if crossover_cycle:
        print(f"  Transfer MAPE < 2% first at cycle {crossover_cycle}")
    else:
        print("  Transfer MAPE never drops below 2% on eval set")

    # Save
    transfer_df.to_csv(os.path.join(RESULTS_DIR, "tl_transfer_predictions.csv"), index=False)
    scratch_df.to_csv(os.path.join(RESULTS_DIR,  "tl_scratch_predictions.csv"),  index=False)
    mape_t.to_csv(os.path.join(RESULTS_DIR, "tl_mape_transfer.csv"), header=["mape"])
    mape_s.to_csv(os.path.join(RESULTS_DIR, "tl_mape_scratch.csv"),  header=["mape"])

    return {
        "transfer_df": transfer_df,
        "scratch_df": scratch_df,
        "mape_transfer": mape_t,
        "mape_scratch": mape_s,
        "crossover_cycle": crossover_cycle,
        "mae_transfer": mae_t,
        "mae_scratch": mae_s,
        "n_seed": n_seed,
        "source_params": source_params,
    }


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Training source GPR on B0005 + B0006 + B0007...")
    _, _, params = train_source_gpr()

    print("\nFitting transfer GPR on B0018 (first 10 cycles)...")
    results = fit_transfer_gpr(target_id="B0018", n_seed=10, source_params=params)
    print(f"\nDone. Crossover cycle: {results['crossover_cycle']}")
