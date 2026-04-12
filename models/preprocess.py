"""
Battery SOH Preprocessing
- Loads discharge cycles from metadata + data files
- Computes SOH = Capacity / nominal_capacity
- Extracts cycle-level features from raw time-series
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_dataset")
NOMINAL_CAPACITY = 2.0  # Ah — standard NASA dataset nominal


def load_soh_curves(battery_ids=None):
    """Return a dict of {battery_id: DataFrame(cycle, capacity, soh)}."""
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    meta["Capacity"] = pd.to_numeric(meta["Capacity"], errors="coerce")
    dis = meta[meta["type"] == "discharge"].copy()

    if battery_ids:
        dis = dis[dis["battery_id"].isin(battery_ids)]

    results = {}
    for bid, grp in dis.groupby("battery_id"):
        grp = grp.dropna(subset=["Capacity"]).reset_index(drop=True)
        grp["cycle"] = np.arange(1, len(grp) + 1)
        grp["soh"] = grp["Capacity"] / NOMINAL_CAPACITY
        results[bid] = grp[["cycle", "Capacity", "soh", "filename"]].copy()

    return results


def extract_cycle_features(battery_ids=None):
    """
    For each discharge cycle, extract scalar features from the raw time-series:
    - mean/std of voltage, current, temperature
    - discharge duration
    - voltage drop (max - min)
    Returns a single DataFrame with columns: battery_id, cycle, soh, + features
    """
    soh_curves = load_soh_curves(battery_ids)
    data_dir = os.path.join(DATA_DIR, "data")

    rows = []
    for bid, df in soh_curves.items():
        print(f"  Extracting features for {bid} ({len(df)} cycles)...")
        for _, row in df.iterrows():
            fpath = os.path.join(data_dir, row["filename"])
            try:
                ts = pd.read_csv(fpath)
            except Exception:
                continue

            # Basic guard
            if ts.empty or "Voltage_measured" not in ts.columns:
                continue

            v = ts["Voltage_measured"].dropna()
            i = ts["Current_measured"].dropna() if "Current_measured" in ts.columns else pd.Series(dtype=float)
            t = ts["Temperature_measured"].dropna() if "Temperature_measured" in ts.columns else pd.Series(dtype=float)
            time = ts["Time"].dropna() if "Time" in ts.columns else pd.Series(dtype=float)

            feat = {
                "battery_id": bid,
                "cycle": row["cycle"],
                "soh": row["soh"],
                "capacity": row["Capacity"],
                "v_mean": v.mean(),
                "v_std": v.std(),
                "v_min": v.min(),
                "v_max": v.max(),
                "v_drop": v.max() - v.min(),
                "i_mean": i.mean() if len(i) > 0 else np.nan,
                "i_std": i.std() if len(i) > 0 else np.nan,
                "i_skew": float(skew(i)) if len(i) > 2 else np.nan,
                "i_kurt": float(kurtosis(i)) if len(i) > 2 else np.nan,
                "temp_mean": t.mean() if len(t) > 0 else np.nan,
                "temp_max": t.max() if len(t) > 0 else np.nan,
                "duration": time.max() - time.min() if len(time) > 1 else np.nan,
            }
            rows.append(feat)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading SOH curves...")
    soh_curves = load_soh_curves(["B0005", "B0006", "B0007", "B0018"])
    for bid, df in soh_curves.items():
        print(f"  {bid}: {len(df)} cycles, SOH {df['soh'].iloc[0]:.3f} → {df['soh'].iloc[-1]:.3f}")

    print("\nExtracting cycle features...")
    features = extract_cycle_features(["B0005", "B0006", "B0007", "B0018"])
    features.to_csv(os.path.join(out_dir, "cycle_features.csv"), index=False)
    print(f"\nSaved {len(features)} rows to results/cycle_features.csv")
    print(features.head())
