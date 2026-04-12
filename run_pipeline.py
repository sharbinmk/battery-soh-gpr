"""
Full pipeline: preprocess → train GPR → generate plots
Run: python3 run_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    from preprocess import load_soh_curves, extract_cycle_features
    from gpr_model import train_gpr_per_battery, train_gpr_multifeature
    from visualize import (
        plot_soh_predictions,
        plot_degradation_comparison,
        plot_uncertainty_vs_cycle,
        plot_error_distribution,
    )

    # ── Step 1: Load SOH curves ───────────────────────────────────────────────
    print("=" * 50)
    print("Step 1: Loading SOH curves")
    print("=" * 50)
    soh_curves = load_soh_curves(BATTERIES)
    for bid, df in soh_curves.items():
        print(f"  {bid}: {len(df)} discharge cycles, SOH {df['soh'].iloc[0]:.3f}→{df['soh'].iloc[-1]:.3f}")

    # ── Step 2: Train GPR per battery ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 2: Training GPR per battery")
    print("=" * 50)
    trained = train_gpr_per_battery(soh_curves)

    # ── Step 3: Extract features + multi-feature GPR ──────────────────────────
    feat_path = os.path.join(RESULTS_DIR, "cycle_features.csv")
    print("\n" + "=" * 50)
    print("Step 3: Extracting cycle features")
    print("=" * 50)
    if os.path.exists(feat_path):
        import pandas as pd
        features = pd.read_csv(feat_path)
        print(f"  Loaded cached features: {len(features)} rows")
    else:
        features = extract_cycle_features(BATTERIES)
        features.to_csv(feat_path, index=False)
        print(f"  Saved {len(features)} rows to results/cycle_features.csv")

    print("\n" + "=" * 50)
    print("Step 4: Multi-feature GPR (leave-one-battery-out)")
    print("=" * 50)
    results_df, metrics_df = train_gpr_multifeature(features)
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    # ── Step 4: Visualize ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 5: Generating plots")
    print("=" * 50)
    plot_soh_predictions()
    plot_degradation_comparison()
    plot_uncertainty_vs_cycle()
    plot_error_distribution()

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("  Next steps:")
    print("    Streamlit dashboard : streamlit run dashboard/app.py")
    print("    FastAPI server      : uvicorn dashboard.api:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    main()
