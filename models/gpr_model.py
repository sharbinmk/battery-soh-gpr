"""
Gaussian Process Regression for Battery SOH Estimation

- Trains a GPR per battery using cycle number as input
- Also supports multi-feature GPR using extracted cycle features
- Outputs predictions with uncertainty bounds
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")


def build_kernel():
    """Matern kernel + noise — well-suited for smooth but non-stationary degradation."""
    return Matern(length_scale=50.0, nu=2.5) + WhiteKernel(noise_level=1e-3)


def train_gpr_per_battery(soh_curves: dict, test_size=0.2, random_state=42):
    """
    Train one GPR per battery using [cycle] → SOH.
    Returns dict of {battery_id: (model, scaler_X, results_df)}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    trained = {}

    for bid, df in soh_curves.items():
        df = df.dropna(subset=["soh"]).sort_values("cycle").reset_index(drop=True)
        X = df[["cycle"]].values.astype(float)
        y = df["soh"].values.astype(float)

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(X)), test_size=test_size, random_state=random_state, shuffle=False
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_all_s = scaler.transform(X)

        gpr = GaussianProcessRegressor(
            kernel=build_kernel(),
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-6,
        )
        gpr.fit(X_train_s, y_train)

        y_pred_all, y_std_all = gpr.predict(X_all_s, return_std=True)

        mae = mean_absolute_error(y_test, gpr.predict(X_test_s))
        rmse = mean_squared_error(y_test, gpr.predict(X_test_s)) ** 0.5

        results = df[["cycle", "soh", "Capacity"]].copy()
        results["soh_pred"] = y_pred_all
        results["soh_std"] = y_std_all
        results["split"] = "train"
        results.loc[idx_test, "split"] = "test"

        print(f"  {bid}: MAE={mae:.4f}  RMSE={rmse:.4f}  kernel={gpr.kernel_}")

        # Save model + scaler
        joblib.dump({"model": gpr, "scaler": scaler}, os.path.join(MODELS_DIR, f"{bid}_gpr.pkl"))
        results.to_csv(os.path.join(RESULTS_DIR, f"{bid}_predictions.csv"), index=False)

        trained[bid] = {"model": gpr, "scaler": scaler, "results": results, "mae": mae, "rmse": rmse}

    return trained


def train_gpr_multifeature(features_df: pd.DataFrame, feature_cols=None, test_size=0.2):
    """
    Train a single GPR on all batteries using cycle features.
    Uses leave-one-battery-out split for evaluation.
    """
    if feature_cols is None:
        feature_cols = ["cycle", "v_mean", "v_std", "v_drop", "temp_mean", "duration"]

    # Drop rows with NaN in feature cols
    df = features_df.dropna(subset=feature_cols + ["soh"]).copy()

    all_results = []
    per_battery_metrics = []

    for bid in df["battery_id"].unique():
        test_mask = df["battery_id"] == bid
        train_df = df[~test_mask]
        test_df = df[test_mask]

        X_train = train_df[feature_cols].values
        y_train = train_df["soh"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["soh"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        gpr = GaussianProcessRegressor(
            kernel=build_kernel(),
            n_restarts_optimizer=3,
            normalize_y=True,
            alpha=1e-6,
        )
        gpr.fit(X_train_s, y_train)

        y_pred, y_std = gpr.predict(X_test_s, return_std=True)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        tmp = test_df[["battery_id", "cycle", "soh", "capacity"]].copy()
        tmp["soh_pred"] = y_pred
        tmp["soh_std"] = y_std
        all_results.append(tmp)

        per_battery_metrics.append({"battery_id": bid, "mae": mae, "rmse": rmse})
        print(f"  Leave-one-out {bid}: MAE={mae:.4f}  RMSE={rmse:.4f}")

    results_df = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(per_battery_metrics)

    results_df.to_csv(os.path.join(RESULTS_DIR, "multifeature_predictions.csv"), index=False)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "multifeature_metrics.csv"), index=False)

    return results_df, metrics_df


def load_model(battery_id: str):
    """Load a saved GPR model for a given battery."""
    path = os.path.join(MODELS_DIR, f"{battery_id}_gpr.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model for {battery_id}")
    return joblib.load(path)


def predict_soh(battery_id: str, cycles: list):
    """Predict SOH + uncertainty for given cycle numbers using saved model."""
    obj = load_model(battery_id)
    gpr, scaler = obj["model"], obj["scaler"]
    X = scaler.transform(np.array(cycles).reshape(-1, 1))
    soh_pred, soh_std = gpr.predict(X, return_std=True)
    return soh_pred.tolist(), soh_std.tolist()


if __name__ == "__main__":
    from preprocess import load_soh_curves, extract_cycle_features

    os.makedirs(RESULTS_DIR, exist_ok=True)

    BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

    print("=== GPR per battery (cycle → SOH) ===")
    soh_curves = load_soh_curves(BATTERIES)
    trained = train_gpr_per_battery(soh_curves)

    print("\n=== Multi-feature GPR (leave-one-out) ===")
    feat_path = os.path.join(RESULTS_DIR, "cycle_features.csv")
    if os.path.exists(feat_path):
        features = pd.read_csv(feat_path)
    else:
        print("  Feature file not found, extracting...")
        features = extract_cycle_features(BATTERIES)
        features.to_csv(feat_path, index=False)

    results_df, metrics_df = train_gpr_multifeature(features)
    print("\nMulti-feature metrics:")
    print(metrics_df.to_string(index=False))
