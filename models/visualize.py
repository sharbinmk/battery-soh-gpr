"""
Visualization utilities for battery SOH GPR results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PALETTE = {"B0005": "#1f77b4", "B0006": "#ff7f0e", "B0007": "#2ca02c", "B0018": "#d62728"}


def plot_soh_predictions(battery_ids=None, save=True):
    """Plot actual vs predicted SOH with uncertainty bands per battery."""
    if battery_ids is None:
        battery_ids = ["B0005", "B0006", "B0007", "B0018"]

    n = len(battery_ids)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, bid in zip(axes, battery_ids):
        path = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
        if not os.path.exists(path):
            ax.set_title(f"{bid} — no data")
            continue

        df = pd.read_csv(path)
        color = PALETTE.get(bid, "steelblue")

        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]

        ax.plot(df["cycle"], df["soh"], "k.", markersize=3, alpha=0.5, label="Actual")
        ax.plot(df["cycle"], df["soh_pred"], color=color, lw=2, label="GPR mean")
        ax.fill_between(
            df["cycle"],
            df["soh_pred"] - 2 * df["soh_std"],
            df["soh_pred"] + 2 * df["soh_std"],
            alpha=0.25, color=color, label="±2σ"
        )
        # Mark train/test split
        if len(test) > 0:
            ax.axvline(test["cycle"].iloc[0], color="gray", linestyle="--", alpha=0.6, label="Train/Test split")

        # EOL threshold at 80% SOH
        ax.axhline(0.8, color="red", linestyle=":", alpha=0.7, label="EOL (80%)")

        ax.set_title(f"{bid}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH")
        ax.set_ylim(0.5, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Battery SOH — Gaussian Process Regression", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "soh_predictions.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    return fig


def plot_degradation_comparison(save=True):
    """Overlay all batteries' SOH curves in one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for bid, color in PALETTE.items():
        path = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df["cycle"], df["soh"], ".", color=color, markersize=2, alpha=0.4)
        ax.plot(df["cycle"], df["soh_pred"], color=color, lw=2, label=bid)
        ax.fill_between(
            df["cycle"],
            df["soh_pred"] - 2 * df["soh_std"],
            df["soh_pred"] + 2 * df["soh_std"],
            alpha=0.15, color=color
        )

    ax.axhline(0.8, color="red", linestyle=":", lw=1.5, label="EOL (80%)")
    ax.set_xlabel("Cycle Number", fontsize=12)
    ax.set_ylabel("State of Health (SOH)", fontsize=12)
    ax.set_title("Battery Degradation Comparison — GPR", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "degradation_comparison.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    return fig


def plot_uncertainty_vs_cycle(save=True):
    """Show how prediction uncertainty grows over cycle number."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for bid, color in PALETTE.items():
        path = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df["cycle"], df["soh_std"], color=color, lw=1.5, label=bid)

    ax.set_xlabel("Cycle Number", fontsize=12)
    ax.set_ylabel("Prediction Std (σ)", fontsize=12)
    ax.set_title("GPR Prediction Uncertainty vs Cycle", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "uncertainty_vs_cycle.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    return fig


def plot_error_distribution(save=True):
    """Histogram of prediction errors per battery."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for ax, (bid, color) in zip(axes, PALETTE.items()):
        path = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        errors = df["soh"] - df["soh_pred"]
        mae = errors.abs().mean()
        ax.hist(errors, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{bid}\nMAE={mae:.4f}", fontsize=11)
        ax.set_xlabel("Error (actual − pred)")

    axes[0].set_ylabel("Count")
    fig.suptitle("Prediction Error Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "error_distribution.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    return fig


if __name__ == "__main__":
    print("Generating plots...")
    plot_soh_predictions()
    plot_degradation_comparison()
    plot_uncertainty_vs_cycle()
    plot_error_distribution()
    print("Done.")
