"""
Streamlit dashboard for Battery SOH GPR results.
Run: streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Allow imports from models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PALETTE = {"B0005": "#1f77b4", "B0006": "#ff7f0e", "B0007": "#2ca02c", "B0018": "#d62728"}
BATTERIES = list(PALETTE.keys())


def load_predictions(bid):
    path = os.path.join(RESULTS_DIR, f"{bid}_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery SOH Monitor",
    page_icon="🔋",
    layout="wide",
)

st.title("🔋 Battery State of Health — GPR Dashboard")
st.markdown("NASA Li-ion Battery Dataset · Gaussian Process Regression with uncertainty quantification")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
selected = st.sidebar.multiselect("Select batteries", BATTERIES, default=BATTERIES)
show_uncertainty = st.sidebar.checkbox("Show ±2σ uncertainty band", value=True)
eol_threshold = st.sidebar.slider("EOL threshold (% SOH)", 60, 90, 80) / 100
show_raw = st.sidebar.checkbox("Show raw measurements", value=True)

# ── Metrics row ───────────────────────────────────────────────────────────────
st.subheader("Summary Metrics")
cols = st.columns(len(selected) if selected else 1)

for col, bid in zip(cols, selected):
    df = load_predictions(bid)
    if df is None:
        col.metric(bid, "No data")
        continue
    current_soh = df["soh"].iloc[-1]
    pred_soh = df["soh_pred"].iloc[-1]
    mae = (df["soh"] - df["soh_pred"]).abs().mean()
    cycles = df["cycle"].max()

    # RUL: first future cycle where predicted SOH drops below threshold
    current_pred_soh = df["soh_pred"].iloc[-1]
    if current_pred_soh <= eol_threshold:
        rul_str = "EOL reached"
    else:
        future = df[df["soh_pred"] <= eol_threshold]
        if len(future) > 0:
            rul = int(future["cycle"].iloc[0] - cycles)
            rul_str = f"{max(rul, 0)} cycles"
        else:
            rul_str = f">{cycles} cycles"

    col.metric(f"🔋 {bid}", f"SOH {current_pred_soh:.1%}", f"MAE {mae:.4f}")
    col.caption(f"Cycles: {cycles} | RUL est.: {rul_str}")

st.divider()

# ── Main plot ─────────────────────────────────────────────────────────────────
st.subheader("SOH Degradation Curves")

fig, ax = plt.subplots(figsize=(12, 5))

for bid in selected:
    df = load_predictions(bid)
    if df is None:
        continue
    color = PALETTE[bid]
    if show_raw:
        ax.plot(df["cycle"], df["soh"], ".", color=color, markersize=2, alpha=0.35)
    ax.plot(df["cycle"], df["soh_pred"], color=color, lw=2, label=bid)
    if show_uncertainty:
        ax.fill_between(
            df["cycle"],
            df["soh_pred"] - 2 * df["soh_std"],
            df["soh_pred"] + 2 * df["soh_std"],
            alpha=0.18, color=color,
        )

ax.axhline(eol_threshold, color="red", linestyle="--", lw=1.5, label=f"EOL ({eol_threshold:.0%})")
ax.set_xlabel("Cycle Number", fontsize=12)
ax.set_ylabel("State of Health (SOH)", fontsize=12)
ax.set_ylim(0.5, 1.1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Per-battery detail ────────────────────────────────────────────────────────
st.subheader("Per-Battery Detail")
tab_labels = [b for b in selected if load_predictions(b) is not None]

if tab_labels:
    tabs = st.tabs(tab_labels)
    for tab, bid in zip(tabs, tab_labels):
        with tab:
            df = load_predictions(bid)
            color = PALETTE[bid]

            c1, c2 = st.columns(2)

            # Left: SOH + uncertainty
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]
            ax2.plot(df["cycle"], df["soh"], "k.", markersize=3, alpha=0.4, label="Actual")
            ax2.plot(df["cycle"], df["soh_pred"], color=color, lw=2, label="GPR mean")
            ax2.fill_between(
                df["cycle"],
                df["soh_pred"] - 2 * df["soh_std"],
                df["soh_pred"] + 2 * df["soh_std"],
                alpha=0.25, color=color, label="±2σ"
            )
            if len(test) > 0:
                ax2.axvline(test["cycle"].iloc[0], color="gray", ls="--", alpha=0.6, label="Train/Test")
            ax2.axhline(eol_threshold, color="red", ls=":", label=f"EOL ({eol_threshold:.0%})")
            ax2.set_xlabel("Cycle")
            ax2.set_ylabel("SOH")
            ax2.set_ylim(0.5, 1.1)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_title(f"{bid} — SOH Prediction")
            plt.tight_layout()
            c1.pyplot(fig2)
            plt.close()

            # Right: Prediction uncertainty
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(df["cycle"], df["soh_std"], color=color, lw=1.5)
            ax3.fill_between(df["cycle"], 0, df["soh_std"], alpha=0.2, color=color)
            ax3.set_xlabel("Cycle")
            ax3.set_ylabel("Std σ")
            ax3.set_title(f"{bid} — Prediction Uncertainty")
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            c2.pyplot(fig3)
            plt.close()

            # Error stats
            errors = df["soh"] - df["soh_pred"]
            st.markdown(f"**Error stats** — MAE: `{errors.abs().mean():.5f}` | RMSE: `{(errors**2).mean()**0.5:.5f}` | Max: `{errors.abs().max():.5f}`")

            # Raw table
            with st.expander("Raw prediction data"):
                st.dataframe(df.style.format({
                    "soh": "{:.4f}", "soh_pred": "{:.4f}", "soh_std": "{:.5f}", "Capacity": "{:.4f}"
                }), use_container_width=True)

# ── Fault Detection ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Fault Detection")
st.markdown(
    "SOH measurements outside the **85% CI** are flagged. "
    "**Amber** = isolated breach (Common Fault). **Red** = 3+ consecutive breaches (Severe Fault)."
)

# 85% CI multiplier: norm.ppf(0.925) ≈ 1.44
CI_85_K = 1.44

fault_battery = st.selectbox("Battery", BATTERIES, key="fault_battery")
fd_df = load_predictions(fault_battery)

if fd_df is not None:
    fd = fd_df.copy()
    ci_lo = fd["soh_pred"] - CI_85_K * fd["soh_std"]
    ci_hi = fd["soh_pred"] + CI_85_K * fd["soh_std"]
    fd["breach"] = (fd["soh"] < ci_lo) | (fd["soh"] > ci_hi)

    # Label consecutive runs of breaches
    def label_faults(breach_series):
        labels = []
        run = 0
        for b in breach_series:
            if b:
                run += 1
            else:
                run = 0
            labels.append(run)
        return labels

    fd["run_len"] = label_faults(fd["breach"].tolist())
    # Back-fill run length to start of each run so we can mark entire runs ≥3 as severe
    # Mark severe: any cycle that is part of a run that reaches length ≥3
    severe_starts = set(fd.index[fd["run_len"] == 3])
    severe_cycles = set()
    for idx in severe_starts:
        severe_cycles.update(range(idx - 2, idx + 1))
        # extend forward while still breaching
        j = idx + 1
        while j < len(fd) and fd.loc[j, "breach"]:
            severe_cycles.add(j)
            j += 1

    fd["fault_type"] = "None"
    fd.loc[fd["breach"], "fault_type"] = "Common"
    fd.loc[fd.index.isin(severe_cycles), "fault_type"] = "Severe"

    # Chart
    fig_fd, ax_fd = plt.subplots(figsize=(12, 4))
    color = PALETTE[fault_battery]

    ax_fd.plot(fd["cycle"], fd["soh"], "k.", markersize=3, alpha=0.45, label="Actual SOH")
    ax_fd.plot(fd["cycle"], fd["soh_pred"], color=color, lw=2, label="GPR mean")
    ax_fd.fill_between(fd["cycle"], ci_lo, ci_hi, alpha=0.2, color=color, label="85% CI")
    ax_fd.axhline(eol_threshold, color="red", linestyle="--", lw=1, alpha=0.6, label=f"EOL ({eol_threshold:.0%})")

    # Scatter fault points
    common = fd[fd["fault_type"] == "Common"]
    severe = fd[fd["fault_type"] == "Severe"]
    if len(common):
        ax_fd.scatter(common["cycle"], common["soh"], color="#FFA500", s=30, zorder=5, label=f"Common Fault ({len(common)})")
    if len(severe):
        ax_fd.scatter(severe["cycle"], severe["soh"], color="red", s=50, marker="X", zorder=6, label=f"Severe Fault ({len(severe)})")

    ax_fd.set_xlabel("Cycle")
    ax_fd.set_ylabel("SOH")
    ax_fd.set_ylim(0.5, 1.1)
    ax_fd.legend(fontsize=9)
    ax_fd.grid(True, alpha=0.3)
    ax_fd.set_title(f"{fault_battery} — Fault Detection (85% CI)")
    plt.tight_layout()
    st.pyplot(fig_fd)
    plt.close()

    # Summary badges
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Total breaches", int(fd["breach"].sum()))
    fc2.metric("Common Faults", int((fd["fault_type"] == "Common").sum()), delta_color="off")
    fc3.metric("Severe Faults", int((fd["fault_type"] == "Severe").sum()), delta_color="inverse")

    # Fault event log
    fault_log = fd[fd["fault_type"] != "None"][["cycle", "soh", "soh_pred", "soh_std", "fault_type"]].copy()
    fault_log.columns = ["Cycle", "Actual SOH", "Pred SOH", "Pred Std", "Fault Type"]

    if len(fault_log):
        def highlight_faults(row):
            if row["Fault Type"] == "Severe":
                return ["background-color: #ffcccc"] * len(row)
            elif row["Fault Type"] == "Common":
                return ["background-color: #fff3cc"] * len(row)
            return [""] * len(row)

        st.markdown("**Fault Event Log**")
        st.dataframe(
            fault_log.style
                .apply(highlight_faults, axis=1)
                .format({"Actual SOH": "{:.4f}", "Pred SOH": "{:.4f}", "Pred Std": "{:.5f}"}),
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No faults detected for this battery.")
else:
    st.warning("No prediction data found. Run the training pipeline first.")

# ── Transfer Learning ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Transfer Learning")
st.markdown(
    "B0018 predicted using only its **first N seed cycles**, with kernel hyperparameters "
    "transferred from B0005 + B0006 + B0007. Compared against training from scratch on the same data."
)

tl_col1, tl_col2 = st.columns([1, 3])
n_seed = tl_col1.slider("Seed cycles (B0018)", min_value=5, max_value=40, value=10, step=1)

if tl_col1.button("Run Transfer Learning", use_container_width=True):
    with st.spinner("Training source GPR and fitting transfer model…"):
        try:
            from transfer_learning import train_source_gpr, fit_transfer_gpr
            import os
            _, _, src_params = train_source_gpr()
            tl = fit_transfer_gpr(target_id="B0018", n_seed=n_seed, source_params=src_params)
            st.session_state["tl_results"] = tl
        except Exception as e:
            st.error(f"Error: {e}")

# Load cached results if available
tl = st.session_state.get("tl_results")
if tl is None:
    # Try loading from saved CSVs
    t_path = os.path.join(RESULTS_DIR, "tl_transfer_predictions.csv")
    s_path = os.path.join(RESULTS_DIR, "tl_scratch_predictions.csv")
    mt_path = os.path.join(RESULTS_DIR, "tl_mape_transfer.csv")
    ms_path = os.path.join(RESULTS_DIR, "tl_mape_scratch.csv")
    if all(os.path.exists(p) for p in [t_path, s_path, mt_path, ms_path]):
        tdf = pd.read_csv(t_path)
        sdf = pd.read_csv(s_path)
        mt  = pd.read_csv(mt_path, index_col=0)["mape"]
        ms  = pd.read_csv(ms_path, index_col=0)["mape"]
        eval_actual = tdf[tdf["region"] == "eval"]["soh"].values
        eval_pred_t = tdf[tdf["region"] == "eval"]["soh_pred"].values
        eval_pred_s = sdf[sdf["region"] == "eval"]["soh_pred"].values
        from sklearn.metrics import mean_absolute_error
        below2 = mt[mt < 2.0]
        tl = {
            "transfer_df": tdf, "scratch_df": sdf,
            "mape_transfer": mt, "mape_scratch": ms,
            "crossover_cycle": int(below2.index[0]) if len(below2) > 0 else None,
            "mae_transfer": mean_absolute_error(eval_actual, eval_pred_t),
            "mae_scratch":  mean_absolute_error(eval_actual, eval_pred_s),
            "n_seed": int((tdf["region"] == "seed").sum()),
            "source_params": {},
        }

if tl:
    t_df = tl["transfer_df"]
    s_df = tl["scratch_df"]
    seed_mask = t_df["region"] == "seed"
    n_used = int(seed_mask.sum())

    # Metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Transfer MAE (eval)", f"{tl['mae_transfer']:.5f}")
    mc2.metric("Scratch MAE (eval)",  f"{tl['mae_scratch']:.5f}",
               delta=f"{tl['mae_scratch'] - tl['mae_transfer']:+.5f}",
               delta_color="inverse")
    mc3.metric("MAPE < 2% at cycle",
               str(tl["crossover_cycle"]) if tl["crossover_cycle"] else "Never")

    # SOH prediction comparison
    fig_tl, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(t_df["cycle"], t_df["soh"], "k.", markersize=3, alpha=0.4, label="Actual")
    ax1.plot(t_df["cycle"], t_df["soh_pred"], "#1f77b4", lw=2, label=f"Transfer (n={n_used})")
    ax1.fill_between(t_df["cycle"],
                     t_df["soh_pred"] - 2 * t_df["soh_std"],
                     t_df["soh_pred"] + 2 * t_df["soh_std"],
                     alpha=0.2, color="#1f77b4")
    ax1.plot(s_df["cycle"], s_df["soh_pred"], "#ff7f0e", lw=2, linestyle="--", label="Scratch")
    ax1.fill_between(s_df["cycle"],
                     s_df["soh_pred"] - 2 * s_df["soh_std"],
                     s_df["soh_pred"] + 2 * s_df["soh_std"],
                     alpha=0.12, color="#ff7f0e")
    # Mark seed region
    seed_end_cycle = t_df[seed_mask]["cycle"].max()
    ax1.axvspan(t_df["cycle"].min(), seed_end_cycle, alpha=0.08, color="green", label=f"Seed ({n_used} cycles)")
    ax1.axhline(eol_threshold, color="red", ls=":", lw=1, label=f"EOL ({eol_threshold:.0%})")
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("SOH")
    ax1.set_ylim(0.5, 1.1); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_title("B0018 — Transfer vs Scratch (SOH)")

    # MAPE comparison
    ax2.plot(tl["mape_transfer"].index, tl["mape_transfer"].values,
             "#1f77b4", lw=2, label="Transfer MAPE")
    ax2.plot(tl["mape_scratch"].index, tl["mape_scratch"].values,
             "#ff7f0e", lw=2, ls="--", label="Scratch MAPE")
    ax2.axhline(2.0, color="green", ls=":", lw=1.5, label="2% threshold")
    if tl["crossover_cycle"]:
        ax2.axvline(tl["crossover_cycle"], color="green", ls="--", alpha=0.7,
                    label=f"Crossover @ cycle {tl['crossover_cycle']}")
    ax2.set_xlabel("Cycle"); ax2.set_ylabel("MAPE (%)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_title("B0018 — MAPE vs Cycle (eval set)")

    plt.tight_layout()
    st.pyplot(fig_tl)
    plt.close()
else:
    st.info("Click **Run Transfer Learning** to generate results.")

# ── Model Comparison ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Comparison")
st.markdown("MAE and RMSE for **GPR**, **Linear Regression**, and **LSTM** across all batteries (80/20 split, cycle + signal features).")

mc_path = os.path.join(RESULTS_DIR, "model_comparison.csv")

mc_col1, mc_col2 = st.columns([1, 3])
if mc_col1.button("Run Model Comparison", use_container_width=True):
    with st.spinner("Training GPR, Linear Regression, and LSTM on all batteries…"):
        try:
            from model_comparison import run_comparison
            mc_results = run_comparison(lstm_epochs=200)
            st.session_state["mc_results"] = mc_results
            mc_path_new = os.path.join(RESULTS_DIR, "model_comparison.csv")
            mc_results.to_csv(mc_path_new, index=False)
        except Exception as e:
            st.error(f"Error: {e}")

mc_df = st.session_state.get("mc_results")
if mc_df is None and os.path.exists(mc_path):
    mc_df = pd.read_csv(mc_path)

if mc_df is not None:
    MODEL_ORDER = ["GPR", "Linear Regression", "LSTM"]
    COLORS = {"GPR": "#1f77b4", "Linear Regression": "#ff7f0e", "LSTM": "#2ca02c"}

    # ── Summary table ─────────────────────────────────────────────────────────
    pivot = mc_df.pivot_table(index="battery_id", columns="model",
                              values=["mae", "rmse"]).round(5)
    # Flatten multi-index columns: "mae GPR", "mae LSTM", ...
    pivot.columns = [f"{m} {b}" for m, b in pivot.columns]
    pivot = pivot.reset_index()

    # Reorder columns: battery | GPR MAE | GPR RMSE | LR MAE | LR RMSE | LSTM MAE | LSTM RMSE
    ordered_cols = ["battery_id"]
    for model in MODEL_ORDER:
        for metric in ["mae", "rmse"]:
            col = f"{metric} {model}"
            if col in pivot.columns:
                ordered_cols.append(col)
    pivot = pivot[ordered_cols]
    pivot.columns = ["Battery"] + [
        c.replace("mae ", "").replace("rmse ", "").replace(" ", " ")
        for c in ordered_cols[1:]
    ]
    # Rebuild cleaner header
    display_cols = ["Battery"]
    for model in MODEL_ORDER:
        display_cols += [f"{model} MAE", f"{model} RMSE"]
    pivot.columns = display_cols[:len(pivot.columns)]

    def highlight_best(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        metric_cols = [c for c in df.columns if c != "Battery"]
        # Group by metric type (MAE / RMSE) across models
        for metric in ["MAE", "RMSE"]:
            cols = [c for c in metric_cols if metric in c]
            if not cols:
                continue
            for i in df.index:
                vals = df.loc[i, cols].astype(float)
                best_col = vals.idxmin()
                styles.loc[i, best_col] = "background-color: #d4edda; font-weight: bold"
        return styles

    st.dataframe(
        pivot.style.apply(highlight_best, axis=None).format(
            {c: "{:.5f}" for c in pivot.columns if c != "Battery"}
        ),
        use_container_width=True,
    )
    st.caption("Green = best (lowest) MAE / RMSE per battery.")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig_mc, axes_mc = plt.subplots(1, 2, figsize=(13, 4))
    batteries = mc_df["battery_id"].unique()
    x = np.arange(len(batteries))
    width = 0.25

    for ax_mc, metric in zip(axes_mc, ["mae", "rmse"]):
        for i, model in enumerate(MODEL_ORDER):
            vals = [
                mc_df[(mc_df["battery_id"] == b) & (mc_df["model"] == model)][metric].values[0]
                if len(mc_df[(mc_df["battery_id"] == b) & (mc_df["model"] == model)]) > 0
                else 0
                for b in batteries
            ]
            ax_mc.bar(x + i * width, vals, width, label=model,
                      color=COLORS[model], alpha=0.85, edgecolor="white")
        ax_mc.set_xticks(x + width)
        ax_mc.set_xticklabels(batteries)
        ax_mc.set_ylabel(metric.upper())
        ax_mc.set_title(f"{metric.upper()} by Model and Battery")
        ax_mc.legend(fontsize=9)
        ax_mc.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_mc)
    plt.close()
else:
    st.info("Click **Run Model Comparison** to train and evaluate all models.")

# ── Feature Ablation ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Feature Ablation: i_skew + i_kurt")
st.markdown(
    "GPR retrained with **skewness** and **kurtosis** of `Current_measured` added alongside "
    "the base feature set. Leave-one-battery-out evaluation. "
    "Negative delta = improvement."
)

abl_path   = os.path.join(RESULTS_DIR, "feature_ablation.csv")
delta_path = os.path.join(RESULTS_DIR, "feature_ablation_delta.csv")

fa_col1, _ = st.columns([1, 3])
if fa_col1.button("Run Feature Ablation", use_container_width=True):
    with st.spinner("Re-extracting features and retraining GPR (base vs extended)…"):
        try:
            from feature_ablation import run_ablation
            abl_result, abl_delta = run_ablation()
            st.session_state["abl_result"] = abl_result
            st.session_state["abl_delta"]  = abl_delta
        except Exception as e:
            st.error(f"Error: {e}")

abl_result = st.session_state.get("abl_result")
abl_delta  = st.session_state.get("abl_delta")
if abl_result is None and os.path.exists(abl_path):
    abl_result = pd.read_csv(abl_path)
if abl_delta is None and os.path.exists(delta_path):
    abl_delta = pd.read_csv(delta_path)

if abl_delta is not None:
    # ── Delta table ───────────────────────────────────────────────────────────
    display_delta = abl_delta.copy()
    display_delta.columns = [
        "Battery", "MAE (base)", "MAE (extended)", "MAE Δ",
        "RMSE (base)", "RMSE (extended)", "RMSE Δ",
    ]

    def highlight_delta(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in ["MAE Δ", "RMSE Δ"]:
            if col not in df.columns:
                continue
            for i in df.index:
                val = df.loc[i, col]
                try:
                    v = float(val)
                    styles.loc[i, col] = (
                        "background-color: #d4edda; color: #155724; font-weight: bold" if v < 0
                        else "background-color: #f8d7da; color: #721c24; font-weight: bold"
                    )
                except (ValueError, TypeError):
                    pass
        return styles

    st.dataframe(
        display_delta.style
            .apply(highlight_delta, axis=None)
            .format({c: "{:.5f}" for c in display_delta.columns if c != "Battery"}),
        use_container_width=True,
    )
    st.caption("Green Δ = MAE/RMSE improved. Red Δ = degraded.")

    # ── Bar chart: MAE base vs extended ───────────────────────────────────────
    if abl_result is not None:
        fig_abl, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(13, 4))
        bats = abl_delta["battery_id"].tolist()
        x_abl = np.arange(len(bats))
        w = 0.35

        for ax_a, metric, base_col, ext_col in [
            (ax_a1, "MAE",  "mae_base",  "mae_extended"),
            (ax_a2, "RMSE", "rmse_base", "rmse_extended"),
        ]:
            ax_a.bar(x_abl - w/2, abl_delta[base_col],  w, label="Base",
                     color="#1f77b4", alpha=0.85, edgecolor="white")
            ax_a.bar(x_abl + w/2, abl_delta[ext_col],   w, label="+ i_skew + i_kurt",
                     color="#2ca02c", alpha=0.85, edgecolor="white")

            # Delta annotation
            for xi, (bv, ev) in enumerate(zip(abl_delta[base_col], abl_delta[ext_col])):
                delta_val = ev - bv
                color = "#155724" if delta_val < 0 else "#721c24"
                ax_a.text(xi + w/2, ev + 0.0003, f"{delta_val:+.4f}",
                          ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

            ax_a.set_xticks(x_abl)
            ax_a.set_xticklabels(bats)
            ax_a.set_ylabel(metric)
            ax_a.set_title(f"{metric}: Base vs Extended Features")
            ax_a.legend(fontsize=9)
            ax_a.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_abl)
        plt.close()

    # ── Verdict ───────────────────────────────────────────────────────────────
    improved = int((abl_delta["mae_delta"] < 0).sum())
    total    = len(abl_delta)
    avg_gain = -abl_delta["mae_delta"].mean() * 100  # as % of 1
    st.info(
        f"**Verdict:** Adding `i_skew` + `i_kurt` improves MAE on **{improved}/{total}** batteries. "
        f"Average MAE reduction: **{avg_gain:.4f}** (absolute)."
    )
else:
    st.info("Click **Run Feature Ablation** to compare base vs extended features.")

# ── Point prediction ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Point Prediction")
st.markdown("Predict SOH at a specific cycle using the saved GPR model.")

col_a, col_b, col_c = st.columns(3)
pred_battery = col_a.selectbox("Battery", BATTERIES)
pred_cycle = col_b.number_input("Cycle number", min_value=1, max_value=500, value=100)

if col_c.button("Predict", use_container_width=True):
    try:
        from gpr_model import predict_soh
        soh_pred, soh_std = predict_soh(pred_battery, [pred_cycle])
        st.success(f"**{pred_battery}** at cycle **{pred_cycle}**: SOH = **{soh_pred[0]:.4f}** ± {2*soh_std[0]:.4f} (95% CI)")
    except FileNotFoundError:
        st.warning("Model not found. Run the training pipeline first.")
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("NASA Li-ion Battery Dataset · GPR via scikit-learn · Dashboard built with Streamlit")
