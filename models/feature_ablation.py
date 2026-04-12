"""
Feature ablation: compare GPR with and without i_skew + i_kurt.
Uses leave-one-battery-out evaluation (same protocol as gpr_model.py).
Saves results/feature_ablation.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BASE_FEATURES    = ["cycle", "v_mean", "v_std", "v_drop", "temp_mean", "duration"]
EXTENDED_FEATURES = BASE_FEATURES + ["i_skew", "i_kurt"]


def _gpr():
    return GaussianProcessRegressor(
        kernel=Matern(length_scale=50.0, nu=2.5) + WhiteKernel(noise_level=1e-3),
        n_restarts_optimizer=3,
        normalize_y=True,
        alpha=1e-6,
    )


def run_ablation(features_df: pd.DataFrame = None):
    """
    Leave-one-battery-out GPR for BASE and EXTENDED feature sets.
    Returns DataFrame with columns: battery_id, feature_set, mae, rmse, mae_delta, rmse_delta
    """
    if features_df is None:
        feat_path = os.path.join(RESULTS_DIR, "cycle_features.csv")
        features_df = pd.read_csv(feat_path)

    records = []

    for label, cols in [("base", BASE_FEATURES), ("extended", EXTENDED_FEATURES)]:
        df = features_df.dropna(subset=cols + ["soh"]).copy()

        for bid in df["battery_id"].unique():
            test_df  = df[df["battery_id"] == bid]
            train_df = df[df["battery_id"] != bid]

            X_tr = train_df[cols].values.astype(float)
            y_tr = train_df["soh"].values.astype(float)
            X_te = test_df[cols].values.astype(float)
            y_te = test_df["soh"].values.astype(float)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            gpr = _gpr()
            gpr.fit(X_tr_s, y_tr)
            preds = gpr.predict(X_te_s)

            records.append({
                "battery_id":  bid,
                "feature_set": label,
                "mae":  mean_absolute_error(y_te, preds),
                "rmse": mean_squared_error(y_te, preds) ** 0.5,
            })
            print(f"  {bid} [{label:8s}]  MAE={records[-1]['mae']:.5f}  RMSE={records[-1]['rmse']:.5f}")

    result = pd.DataFrame(records)

    # Compute per-battery delta (extended - base): negative = improvement
    base_df = result[result["feature_set"] == "base"].set_index("battery_id")
    ext_df  = result[result["feature_set"] == "extended"].set_index("battery_id")

    delta = pd.DataFrame({
        "battery_id": base_df.index,
        "mae_base":      base_df["mae"].values,
        "mae_extended":  ext_df["mae"].values,
        "mae_delta":     ext_df["mae"].values - base_df["mae"].values,
        "rmse_base":     base_df["rmse"].values,
        "rmse_extended": ext_df["rmse"].values,
        "rmse_delta":    ext_df["rmse"].values - base_df["rmse"].values,
    }).reset_index(drop=True)

    result.to_csv(os.path.join(RESULTS_DIR, "feature_ablation.csv"), index=False)
    delta.to_csv(os.path.join(RESULTS_DIR, "feature_ablation_delta.csv"), index=False)
    return result, delta


if __name__ == "__main__":
    print("Running feature ablation (base vs base + i_skew + i_kurt)...\n")
    result, delta = run_ablation()
    print("\nDelta table (negative = improvement):")
    print(delta.to_string(index=False))
