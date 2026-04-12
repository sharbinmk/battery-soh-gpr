"""
Model Comparison: GPR vs Linear Regression vs LSTM
Evaluates all three on each battery using the same 80/20 train-test split.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BATTERIES   = ["B0005", "B0006", "B0007", "B0018"]


# ── LSTM definition ───────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _make_sequences(X, y, seq_len=10):
    """Sliding-window sequences for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_test, y_test,
               seq_len=10, hidden=32, layers=2,
               epochs=200, lr=1e-3, batch=16):
    """Train LSTM, return (mae, rmse) on test set."""
    # Sequences
    X_tr_seq, y_tr_seq = _make_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = _make_sequences(X_test,  y_test,  seq_len)

    if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
        return np.nan, np.nan

    X_tr_t = torch.FloatTensor(X_tr_seq)
    y_tr_t = torch.FloatTensor(y_tr_seq)
    X_te_t = torch.FloatTensor(X_te_seq)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch, shuffle=False)

    model = LSTMRegressor(input_size=X_train.shape[1], hidden_size=hidden, num_layers=layers)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).numpy()

    mae  = mean_absolute_error(y_te_seq, preds)
    rmse = mean_squared_error(y_te_seq, preds) ** 0.5
    return mae, rmse


# ── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(feature_cols=None, test_size=0.2, lstm_epochs=200):
    """
    For each battery, train GPR / LinearRegression / LSTM on the same
    train split and evaluate on the test split.

    Uses cycle_features.csv if available (richer features),
    falls back to cycle-only if not.

    Returns a DataFrame with columns:
      battery_id, model, mae, rmse
    """
    feat_path = os.path.join(RESULTS_DIR, "cycle_features.csv")
    use_features = os.path.exists(feat_path)

    if use_features and feature_cols is None:
        feature_cols = ["cycle", "v_mean", "v_std", "v_drop", "temp_mean", "duration"]
        df_all = pd.read_csv(feat_path).dropna(subset=feature_cols + ["soh"])
    else:
        # Fallback: load from prediction CSVs, cycle only
        feature_cols = ["cycle"]
        rows = []
        for bid in BATTERIES:
            p = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)[["cycle", "soh"]]
                d["battery_id"] = bid
                rows.append(d)
        df_all = pd.concat(rows, ignore_index=True)

    records = []

    for bid in BATTERIES:
        df = df_all[df_all["battery_id"] == bid].sort_values("cycle").reset_index(drop=True)
        if len(df) < 20:
            continue

        split = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

        X_train = train_df[feature_cols].values.astype(float)
        y_train = train_df["soh"].values.astype(float)
        X_test  = test_df[feature_cols].values.astype(float)
        y_test  = test_df["soh"].values.astype(float)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        print(f"  {bid}:", end=" ", flush=True)

        # ── GPR ───────────────────────────────────────────────────────────────
        gpr = GaussianProcessRegressor(
            kernel=Matern(length_scale=50.0, nu=2.5) + WhiteKernel(noise_level=1e-3),
            n_restarts_optimizer=3, normalize_y=True, alpha=1e-6,
        )
        gpr.fit(X_tr_s, y_train)
        gpr_pred = gpr.predict(X_te_s)
        gpr_mae  = mean_absolute_error(y_test, gpr_pred)
        gpr_rmse = mean_squared_error(y_test, gpr_pred) ** 0.5
        records.append({"battery_id": bid, "model": "GPR",
                        "mae": gpr_mae, "rmse": gpr_rmse})
        print("GPR✓", end=" ", flush=True)

        # ── Linear Regression ─────────────────────────────────────────────────
        lr = LinearRegression()
        lr.fit(X_tr_s, y_train)
        lr_pred = lr.predict(X_te_s)
        lr_mae  = mean_absolute_error(y_test, lr_pred)
        lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5
        records.append({"battery_id": bid, "model": "Linear Regression",
                        "mae": lr_mae, "rmse": lr_rmse})
        print("LR✓", end=" ", flush=True)

        # ── LSTM ──────────────────────────────────────────────────────────────
        # Scale to [0,1] for LSTM stability
        lstm_scaler = StandardScaler()
        X_tr_lstm = lstm_scaler.fit_transform(X_train)
        X_te_lstm = lstm_scaler.transform(X_test)
        lstm_mae, lstm_rmse = train_lstm(
            X_tr_lstm, y_train, X_te_lstm, y_test,
            seq_len=min(10, split // 3), epochs=lstm_epochs,
        )
        records.append({"battery_id": bid, "model": "LSTM",
                        "mae": lstm_mae, "rmse": lstm_rmse})
        print("LSTM✓")

    results = pd.DataFrame(records)
    results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    return results


def pivot_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Return a wide table: rows = battery × model, columns = MAE / RMSE."""
    return results.pivot_table(
        index="battery_id", columns="model", values=["mae", "rmse"]
    ).round(5)


if __name__ == "__main__":
    print("Running model comparison (GPR / LR / LSTM)...")
    results = run_comparison()
    print("\nResults:")
    print(pivot_comparison(results).to_string())
