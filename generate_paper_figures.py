# -*- coding: utf-8 -*-
"""Generate publication-quality figures from W1+W2+LoRA CSV results (0 CU, local CPU)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TAB_DIR = ROOT / "results" / "tables"
if not TAB_DIR.exists():
    TAB_DIR = ROOT / "_colab_outputs_w2" / "outputs" / "tables"
OUT_DIR = ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

MODEL_ORDER = ["one_hot", "esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"]
MODEL_LABELS = {"one_hot": "One-Hot", "esm2_8m": "8M", "esm2_35m": "35M",
                "esm2_150m": "150M", "esm2_650m": "650M", "esm2_3b": "3B"}
MODEL_COLORS = {
    "one_hot": "#6c757d",
    "esm2_8m": "#4c72b0",
    "esm2_35m": "#55a868",
    "esm2_150m": "#dd8452",
    "esm2_650m": "#8172b3",
    "esm2_3b": "#c44e52",
}


def make_fig1(ridge_df: pd.DataFrame):
    splits = list(ridge_df.groupby(["dataset", "split"], sort=False).groups.keys())
    ncols = 4
    nrows = int(np.ceil(len(splits) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for idx, (ds, sp) in enumerate(splits):
        ax = axes[idx]
        sub = ridge_df[(ridge_df["dataset"] == ds) & (ridge_df["split"] == sp)].set_index("model").reindex(MODEL_ORDER)
        vals = sub["spearman"].fillna(0.0).values
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
        bars = ax.bar(range(len(MODEL_ORDER)), vals, color=colors, edgecolor="#222222", linewidth=0.5)
        for j, m in enumerate(MODEL_ORDER):
            if m in sub.index and (pd.isna(sub.loc[m, "spearman"]) or bool(sub.loc[m, "pred_constant"])):
                ax.text(j, 0.02, "const", ha="center", va="bottom", fontsize=7.5, color="#b00020", fontweight="bold")
        ax.axhline(0, color="#666666", linewidth=0.6)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right")
        ax.set_title(f"{ds} / {sp}", fontweight="bold", pad=6)
        if idx % ncols == 0:
            ax.set_ylabel("Spearman ρ")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    for k in range(len(splits), len(axes)):
        axes[k].axis("off")
    fig.suptitle("FLIP2 Frozen-Embedding Ridge Probing Across ESM-2 Scales (12 Regression Splits)",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT_DIR / "fig1_model_scale_probing.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_fig2(de_df: pd.DataFrame):
    natural_order = [
        ("amylase", "one_to_many"), ("amylase", "close_to_far"), ("rhomax", "by_wild_type"),
        ("ired", "two_to_many"),
        ("hydro", "three_to_many"), ("hydro", "low_to_high"),
        ("hydro", "to_P06241"), ("hydro", "to_P0A9X9"), ("hydro", "to_P01053"),
        ("trpb", "one_to_many"), ("trpb", "two_to_many"), ("trpb", "by_position"),
    ]
    avail = set(de_df.groupby(["dataset", "split"]).groups.keys())
    splits = [s for s in natural_order if s in avail]
    ncols = 4
    nrows = int(np.ceil(len(splits) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    handles_map = {}
    for idx, (ds, sp) in enumerate(splits):
        ax = axes[idx]
        sub = de_df[(de_df["dataset"] == ds) & (de_df["split"] == sp)]
        for m in MODEL_ORDER:
            gm = sub[sub["model"] == m].sort_values("fraction")
            if gm.empty or gm["spearman_mean"].isna().all():
                continue
            (ln,) = ax.plot(gm["fraction"], gm["spearman_mean"], marker="o", markersize=4, linewidth=1.7,
                            label=MODEL_LABELS[m], color=MODEL_COLORS[m])
            handles_map.setdefault(m, ln)
            if gm["spearman_std"].notna().any():
                sd = gm["spearman_std"].fillna(0.0)
                ax.fill_between(gm["fraction"], gm["spearman_mean"] - sd, gm["spearman_mean"] + sd,
                                color=MODEL_COLORS[m], alpha=0.12)
        ax.set_title(f"{ds} / {sp}", fontweight="bold", pad=6)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Training fraction")
        if idx % ncols == 0:
            ax.set_ylabel("Spearman ρ (mean ± std)")
        ax.grid(alpha=0.25, linestyle="--")
    for k in range(len(splits), len(axes)):
        axes[k].axis("off")
    handles = [handles_map[m] for m in MODEL_ORDER if m in handles_map]
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER if m in handles_map]
    fig.suptitle("Data Efficiency on FLIP2 Distribution-Shift Splits (3 Seeds, Frozen ESM-2 Ridge)",
                 fontsize=13.5, fontweight="bold", y=0.992)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962),
               ncol=6, frameon=True, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(OUT_DIR / "fig2_data_efficiency.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_fig3(cov_df: pd.DataFrame, uq_df: pd.DataFrame):
    pairs = list(cov_df.groupby(["dataset", "split", "model"], sort=False).groups.keys())
    ncols = 4
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    uq_idx = uq_df.set_index(["dataset", "split", "model"])
    for idx, (ds, sp, m) in enumerate(pairs):
        ax = axes[idx]
        sub = cov_df[(cov_df["dataset"] == ds) & (cov_df["split"] == sp) & (cov_df["model"] == m)].sort_values("keep_fraction", ascending=False)
        col = "#8172b3" if m == "esm2_650m" else "#c44e52"
        ax.plot(sub["keep_fraction"], sub["rmse"], marker="o", markersize=4.5, linewidth=1.8, color=col)
        ax.invert_xaxis()
        rho_err = uq_idx.loc[(ds, sp, m), "std_abserr_spearman"]
        ax.set_title(f"{ds}/{sp} ({MODEL_LABELS[m]})\nstd~|err| ρ={rho_err:+.3f}", fontsize=9.5, fontweight="bold", pad=6)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Retained fraction (low → high uncertainty)")
        if idx % ncols == 0:
            ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25, linestyle="--")
    for k in range(len(pairs), len(axes)):
        axes[k].axis("off")
    fig.suptitle("5× Deep Ensemble Selective Prediction Curves (Error vs. Coverage)",
                 fontsize=13.5, fontweight="bold", y=0.992)
    fig.tight_layout(rect=[0, 0, 1, 0.968], h_pad=1.6, w_pad=1.1)
    fig.savefig(OUT_DIR / "fig3_uq_error_coverage.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_fig4(nucb_df: pd.DataFrame):
    best = []
    for m in MODEL_ORDER:
        sub = nucb_df[nucb_df["model"] == m]
        # Primary: raw for one_hot, zscore_fit for PLM classification & l2_norm for ordinal
        row = sub[sub["feature_scaling"] == ("raw" if m == "one_hot" else "zscore_fit")].iloc[0]
        row_ord = sub[sub["feature_scaling"] == ("raw" if m == "one_hot" else "l2_norm")].iloc[0]
        best.append({
            "model": m,
            "auroc": float(row["auroc_bin_ge2"]),
            "f1": float(row["macro_f1_4bin"]),
            "ord_rho": float(row_ord["ordinal_spearman"]),
        })
    bdf = pd.DataFrame(best)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    specs = [
        ("auroc", "High-Activity Detection\nAUROC (bin ≥ 2)"),
        ("f1", "4-Bin Activity Classification\nMacro-F1 (bin 3 unseen in fit)"),
        ("ord_rho", "Ordinal Ranking\nSpearman ρ (L2-norm Ridge)"),
    ]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    for ax, (col, title) in zip(axes, specs):
        bars = ax.bar(range(len(MODEL_ORDER)), bdf[col], color=colors, edgecolor="#222222", linewidth=0.6)
        for b, v in zip(bars, bdf[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_ylim(0, bdf[col].max() * 1.18)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.suptitle("NucB (two_to_many, n_test = 48,304): Frozen ESM-2 Scales Strongly on Unseen High-Activity Discovery",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_nucb_classification.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_fig5_lora(ridge_df: pd.DataFrame, uq_df: pd.DataFrame, lora_sum: pd.DataFrame, lora_seed: pd.DataFrame):
    tasks = [
        ("amylase", "one_to_many", "In-Distribution\n(amylase / one_to_many)"),
        ("amylase", "close_to_far", "Distance Extrapolation\n(amylase / close_to_far)"),
        ("hydro", "three_to_many", "Higher-Order Epistasis\n(hydro / three_to_many)"),
        ("hydro", "to_P0A9X9", "Cross-Homolog Transfer\n(hydro / to_P0A9X9)"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # Panel A: Grouped bar chart comparing One-Hot, Frozen 3B Ridge, Frozen 3B 5xMLP, LoRA 3B
    x = np.arange(len(tasks))
    w = 0.20
    oh_vals, fr_vals, mlp_vals, lora_vals, lora_errs = [], [], [], [], []
    for ds, sp, _ in tasks:
        oh = ridge_df[(ridge_df["dataset"] == ds) & (ridge_df["split"] == sp) & (ridge_df["model"] == "one_hot")].iloc[0]["spearman"]
        oh_vals.append(0.0 if pd.isna(oh) else float(oh))
        fr_vals.append(float(ridge_df[(ridge_df["dataset"] == ds) & (ridge_df["split"] == sp) & (ridge_df["model"] == "esm2_3b")].iloc[0]["spearman"]))
        mlp_vals.append(float(uq_df[(uq_df["dataset"] == ds) & (uq_df["split"] == sp) & (uq_df["model"] == "esm2_3b")].iloc[0]["ensemble_spearman"]))
        lr = lora_sum[(lora_sum["dataset"] == ds) & (lora_sum["split"] == sp)].iloc[0]
        lora_vals.append(float(lr["spearman_mean"]))
        lora_errs.append(float(lr["spearman_std"]))

    b1 = ax1.bar(x - 1.5 * w, oh_vals, width=w, label="One-Hot Ridge", color="#6c757d", edgecolor="#222222", linewidth=0.5)
    b2 = ax1.bar(x - 0.5 * w, fr_vals, width=w, label="Frozen 3B Ridge", color="#4c72b0", edgecolor="#222222", linewidth=0.5)
    b3 = ax1.bar(x + 0.5 * w, mlp_vals, width=w, label="Frozen 3B 5×MLP", color="#8172b3", edgecolor="#222222", linewidth=0.5)
    b4 = ax1.bar(x + 1.5 * w, lora_vals, yerr=lora_errs, capsize=3, width=w, label="LoRA 3B (3 seeds)",
                 color="#c44e52", edgecolor="#222222", linewidth=0.5)

    for j, v in enumerate(oh_vals):
        if tasks[j][1] == "close_to_far":
            ax1.text(x[j] - 1.5 * w, 0.02, "NaN", ha="center", va="bottom", fontsize=7.2, color="#b00020", fontweight="bold")
        else:
            y_pos = v + 0.015 if v >= 0 else v - 0.018
            va = "bottom" if v >= 0 else "top"
            ax1.text(x[j] - 1.5 * w, y_pos, f"{v:.2f}", ha="center", va=va, fontsize=7.2)
    for bars, errs in ((b2, None), (b3, None), (b4, lora_errs)):
        for j, bar in enumerate(bars):
            h = bar.get_height()
            err = errs[j] if errs is not None else 0.0
            y_pos = (h + err + 0.015) if h >= 0 else (h - err - 0.018)
            va = "bottom" if h >= 0 else "top"
            ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{h:.2f}", ha="center", va=va, fontsize=7.2)

    ax1.axhline(0, color="#555555", linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t[2] for t in tasks])
    ax1.set_ylabel("Test Spearman ρ")
    ax1.set_ylim(-0.28, 0.90)
    ax1.set_title("A. Frozen ESM-2 3B vs. End-to-End LoRA Fine-Tuning", fontweight="bold", pad=8)
    ax1.legend(loc="upper right", frameon=True, ncol=2, fontsize=7.8)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel B: Val vs Test Spearman gap explaining OOD overfitting
    val_means = [float(lora_seed[(lora_seed["dataset"] == ds) & (lora_seed["split"] == sp)]["best_val_spearman"].mean()) for ds, sp, _ in tasks]
    val_stds  = [float(lora_seed[(lora_seed["dataset"] == ds) & (lora_seed["split"] == sp)]["best_val_spearman"].std()) for ds, sp, _ in tasks]
    w2 = 0.30
    ax2.bar(x - w2 / 2, val_means, yerr=val_stds, capsize=3, width=w2, label="LoRA Best Val ρ (in-train split)",
            color="#dd8452", edgecolor="#222222", linewidth=0.5)
    ax2.bar(x + w2 / 2, lora_vals, yerr=lora_errs, capsize=3, width=w2, label="LoRA Test ρ (OOD / target)",
            color="#c44e52", edgecolor="#222222", linewidth=0.5)
    for j in range(len(tasks)):
        gap = lora_vals[j] - val_means[j]
        top_y = max(val_means[j] + val_stds[j], lora_vals[j] + lora_errs[j]) + 0.025
        ax2.text(x[j], top_y, f"Δ={gap:+.2f}",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                 color="#1b7837" if gap > -0.05 else "#b00020")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{ds}\n{sp}" for ds, sp, _ in tasks])
    ax2.set_ylabel("Spearman ρ (mean ± std across 3 seeds)")
    ax2.set_ylim(0, 1.06)
    ax2.set_title("B. Why LoRA Degrades OOD: Val-vs-Test Divergence", fontweight="bold", pad=8)
    ax2.legend(loc="upper center", frameon=True, ncol=1, fontsize=7.8)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Fine-Tuning Inversion: LoRA Helps In-Distribution Interpolation (+0.03) but Overfits on Distribution Shifts (−0.18 to −0.30)",
                 fontsize=12.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_lora_vs_frozen_3b.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_fig6_scaling(abl_df: pd.DataFrame):
    panels = [
        ("trpb", "two_to_many", "trpb / two_to_many\n(zscore_fit induces 3B inversion; l2_norm fixes it)"),
        ("rhomax", "by_wild_type", "rhomax / by_wild_type\n(variable-length homologs: raw 650M wins)"),
        ("hydro", "to_P0A9X9", "hydro / to_P0A9X9\n(cross-homolog: l2_norm peaks at 3B)"),
        ("hydro", "three_to_many", "hydro / three_to_many\n(epistasis: l2_norm & raw align cleanly)"),
    ]
    scalings = [("raw", "raw", "#6c757d", "o"),
                ("l2_norm", "l2_norm", "#1b9e77", "s"),
                ("zscore_fit", "zscore_fit", "#d95f02", "^")]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1))
    plm_order = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"]
    for idx, (ax, (ds, sp, title)) in enumerate(zip(axes, panels)):
        sub = abl_df[(abl_df["dataset"] == ds) & (abl_df["split"] == sp)]
        for sc_key, sc_lbl, col, mk in scalings:
            g = sub[sub["feature_scaling"] == sc_key].set_index("model").reindex(plm_order)
            y = g["spearman"].values
            valid = ~np.isnan(y)
            ax.plot(np.arange(len(plm_order))[valid], y[valid], marker=mk, markersize=5.5,
                    linewidth=1.8, color=col, label=sc_lbl)
        ax.set_xticks(range(len(plm_order)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in plm_order])
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=8)
        if idx == 0:
            ax.set_ylabel("Test Spearman ρ")
            ax.legend(loc="lower left", frameon=True)
        ax.grid(alpha=0.25, linestyle="--")
    fig.suptitle("Embedding Normalization Ablation: High-Dimensional Z-Scoring Amplifies Low-Variance Noise Under Shift",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_scaling_ablation_mechanism.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_fig7_direction_f(summary_json_path: Path):
    if not summary_json_path.exists():
        return
    import json
    data = json.loads(summary_json_path.read_text(encoding="utf-8"))
    fams = [
        ("clinvar_coord", "ClinVar\nCoord Extract"),
        ("gnomad_cli", "gnomAD\nRelative CLI"),
        ("cross_db_chain", "ClinVar →\nEnsembl VEP"),
        ("protein_struct", "UniProt →\nAFDB + pLDDT"),
        ("lit_and_go", "arXiv / EPMC /\nQuickGO"),
    ]
    up_vals = [data["upstream"]["by_family"][k]["pass"] / data["upstream"]["by_family"][k]["n"] * 100 for k, _ in fams]
    pa_vals = [data["patched"]["by_family"][k]["pass"] / data["patched"]["by_family"][k]["n"] * 100 for k, _ in fams]
    up_tot = data["upstream"]["pass_rate"] * 100
    pa_tot = data["patched"]["pass_rate"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]})
    x = np.arange(len(fams))
    w = 0.34
    b1 = ax1.bar(x - w / 2, up_vals, width=w, label=f"Upstream v1.4.0 ({up_tot:.0f}% overall)",
                 color="#d95f02", edgecolor="#222222", linewidth=0.6)
    b2 = ax1.bar(x + w / 2, pa_vals, width=w, label=f"Patched + UA ({pa_tot:.0f}% overall)",
                 color="#1b9e77", edgecolor="#222222", linewidth=0.6)
    for bar, v in zip(b1, up_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, v in zip(b2, pa_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in fams])
    ax1.set_ylabel("Task Pass Rate (%)")
    ax1.set_ylim(0, 128)
    ax1.set_title("A. Pass Rate Across 50 Live Multi-Database Tasks", fontweight="bold", pad=8)
    ax1.legend(loc="upper center", ncol=2, frameon=True)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    modes = [
        ("ClinVar silent Gene-span\n& null ref/alt (Issue #15)", 10, "#c44e52"),
        ("Cascading ClinVar→VEP\nchain failure", 10, "#dd8452"),
        ("gnomAD os.makedirs('')\nCLI crash (Issue #16)", 10, "#8172b3"),
        ("AFDB empty User-Agent\nHTTP 403 (Issue #2)", 10, "#4c72b0"),
        ("arXiv concatenated JSON\n(Issue #13)", 4, "#6c757d"),
    ]
    y_pos = np.arange(len(modes))
    counts = [m[1] for m in modes]
    cols = [m[2] for m in modes]
    bars = ax2.barh(y_pos, counts, color=cols, edgecolor="#222222", linewidth=0.6)
    ax2.invert_yaxis()
    for bar, c in zip(bars, counts):
        ax2.text(c + 0.25, bar.get_y() + bar.get_height() / 2, f"{c} tasks", va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m[0] for m in modes])
    ax2.set_xlabel("Failed Tasks in Upstream v1.4.0 (out of 50)")
    ax2.set_xlim(0, 13.0)
    ax2.set_title("B. Root-Cause Breakdown of 44 Upstream Failures", fontweight="bold", pad=8)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")

    fig.suptitle("ScienceBench-Bio50: Empirical Audit of google-deepmind/science-skills v1.4.0 (12% → 90% Pass Rate)",
                 fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_sciencebench_bio50.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ridge_df = pd.read_csv(TAB_DIR / "ridge_probing_metrics.csv")
    abl_df = pd.read_csv(TAB_DIR / "ridge_probing_ablation.csv")
    de_df = pd.read_csv(TAB_DIR / "data_efficiency_summary.csv")
    uq_df = pd.read_csv(TAB_DIR / "uq_ensemble_metrics.csv")
    cov_df = pd.read_csv(TAB_DIR / "uq_error_coverage_curve.csv")
    nucb_df = pd.read_csv(TAB_DIR / "nucb_classification_metrics.csv")
    lora_sum = pd.read_csv(TAB_DIR / "lora_3b_summary.csv")
    lora_seed = pd.read_csv(TAB_DIR / "lora_3b_per_seed.csv")

    make_fig1(ridge_df)
    make_fig2(de_df)
    make_fig3(cov_df, uq_df)
    make_fig4(nucb_df)
    make_fig5_lora(ridge_df, uq_df, lora_sum, lora_seed)
    make_fig6_scaling(abl_df)
    make_fig7_direction_f(ROOT / "direction_f_science_skills" / "results" / "bench50_summary.json")
    print("Generated all 7 publication figures in", OUT_DIR)
