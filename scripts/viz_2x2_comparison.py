#!/usr/bin/env python
"""
Static figures for the 2x2 niche-comparison pipeline ({scVI,scGPT} x {Banksy,STAGATE}).
Reads compare_niches.py / permutation_composition.py outputs and renders three
PNGs: cross-method domain agreement (+k-robustness), cross-region niche matching
(+significance), and the Q1 annotation-axis composition-cosine story.

Palette: fixed categorical order per the project's dataviz palette (see
references/palette.md in the dataviz skill) -- V1=slot1 blue, V2=slot2 orange,
matched=slot3 aqua, scVI=slot4 yellow, scGPT=slot5 magenta. Status colors
(good/critical) are the separate fixed status palette, never reused as series.

Runs in the neuro env (matplotlib/pandas/numpy only).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------- palette (fixed categorical order; see dataviz skill references/palette.md) ----------
C_V1 = "#2a78d6"       # slot 1 blue
C_V2 = "#eb6834"       # slot 2 orange
C_MATCHED = "#1baf7a"  # slot 3 aqua
C_SCVI = "#eda100"     # slot 4 yellow
C_SCGPT = "#e87ba4"    # slot 5 magenta
C_GOOD = "#0ca30c"     # status: good (not reused as a series color)
C_CRITICAL = "#d03b3b"  # status: critical
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_axes(ax, hgrid=True):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1)
    ax.tick_params(length=0)
    if hgrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)


def fig1_cross_method_agreement(ndir: Path, outdir: Path):
    df = pd.read_csv(ndir / "crossmethod_ari_by_k.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    colors = {"v1": C_V1, "v2": C_V2}
    last_vals = df.sort_values("k").groupby("region")["ari"].last()
    labels_collide = abs(float(last_vals.get("v1", 0)) - float(last_vals.get("v2", 0))) < 0.035
    for region, sub in df.groupby("region"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["ari"], color=colors[region], linewidth=2,
                marker="o", markersize=8, markerfacecolor=colors[region],
                markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round")
        if not labels_collide:
            last = sub.iloc[-1]
            ax.annotate(region.upper(), (last["k"], last["ari"]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=10,
                        color=colors[region], fontweight="bold")

    style_axes(ax)
    ax.set_xlabel("Domain count (k)")
    ax.set_ylabel("Adjusted Rand Index (Banksy vs STAGATE)")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.set_ylim(0, max(0.75, df["ari"].max() * 1.15))
    ax.set_title("Cross-method domain agreement is robust across k", loc="left",
                 fontsize=13, fontweight="bold", color=INK, pad=14)
    legend_handles = [Line2D([0], [0], color=colors[r], lw=2, marker="o",
                              markersize=6, label=r.upper()) for r in ["v1", "v2"]]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    p_all_sig = bool((df["p_value_ari"] < 0.05).all())
    n_perm = 1000
    ax.text(0.0, -0.19,
            f"Permutation test: ARI significant (p<0.05, n={n_perm}) at every k "
            f"in {df['k'].min()}-{df['k'].max()}" if p_all_sig else
            "Permutation p-values vary across k (see table)",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(outdir / "1_cross_method_agreement.png", bbox_inches="tight")
    plt.close(fig)


def fig2_cross_region_matching(ndir: Path, perm_dir: Path, outdir: Path, k=8):
    df = pd.read_csv(ndir / "crossregion_match_by_k.csv")
    d8 = df[df["k"] == k].copy()
    method_label = {"banksy": "Banksy", "stagate": "STAGATE"}
    ann_label = {"scvi": "scVI", "scgpt": "scGPT"}
    d8["arm"] = d8["annotation"].map(ann_label) + " x " + d8["method"].map(method_label)
    d8["method"] = pd.Categorical(d8["method"], categories=["banksy", "stagate"], ordered=True)
    d8["annotation"] = pd.Categorical(d8["annotation"], categories=["scvi", "scgpt"], ordered=True)
    d8 = d8.sort_values(["method", "annotation"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200)

    # Panel A: stacked bar of matched / v1-specific / v2-specific at k=8
    ax = axes[0]
    x = np.arange(len(d8))
    bw = 0.5
    bottoms = np.zeros(len(d8))
    segments = [("n_matched", "Matched", C_MATCHED),
                ("n_region0_specific", "V1-specific", C_V1),
                ("n_region1_specific", "V2-specific", C_V2)]
    for col, label, color in segments:
        vals = d8[col].values.astype(float)
        ax.bar(x, vals, width=bw, bottom=bottoms, color=color, label=label,
               edgecolor=SURFACE, linewidth=2)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 1.2:
                ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center",
                        fontsize=9, color="white" if color != C_MATCHED else "white",
                        fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(d8["arm"], fontsize=9)
    style_axes(ax)
    ax.set_ylabel("Niche count")
    ax.set_title("A. Cross-region matching at k=8", loc="left", fontsize=12,
                 fontweight="bold", color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    # Panel B: median permutation -log10(p) per arm, grouped by method, hued by annotation
    files = {
        ("scvi", "banksy"): "banksy_scvi_v1v2_crossregion_k8_pvalues.csv",
        ("scvi", "stagate"): "stagate_scvi_v1v2_crossregion_k8_pvalues.csv",
        ("scgpt", "banksy"): "banksy_scgpt_v1v2_crossregion_k8_pvalues.csv",
        ("scgpt", "stagate"): "stagate_scgpt_v1v2_crossregion_k8_pvalues.csv",
    }
    rows = []
    for (ann, meth), fname in files.items():
        p = pd.read_csv(perm_dir / fname)
        med_p = float(p["p_value"].median())
        rows.append({"annotation": ann, "method": meth, "median_p": med_p})
    pdf = pd.DataFrame(rows)
    pdf["neglog10p"] = -np.log10(pdf["median_p"].clip(lower=1e-4))

    ax = axes[1]
    methods = ["banksy", "stagate"]
    xpos = np.arange(len(methods))
    width = 0.32
    for i, ann in enumerate(["scvi", "scgpt"]):
        sub = pdf[pdf["annotation"] == ann].set_index("method").reindex(methods)
        color = C_SCVI if ann == "scvi" else C_SCGPT
        offset = (i - 0.5) * width
        bars = ax.bar(xpos + offset, sub["neglog10p"], width=width, color=color,
                       label="scVI" if ann == "scvi" else "scGPT",
                       edgecolor=SURFACE, linewidth=1)
        for xb, v, mp in zip(xpos + offset, sub["neglog10p"], sub["median_p"]):
            ax.text(xb, v + 0.08, f"p={mp:.3g}", ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)
    thr = -np.log10(0.05)
    ax.axhline(thr, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(methods) - 0.5, thr + 0.08, "p = 0.05 threshold", fontsize=8,
            color=INK_MUTED, ha="right")
    ax.set_xticks(xpos)
    ax.set_xticklabels([method_label[m] for m in methods])
    style_axes(ax)
    ax.set_ylabel("-log10(median permutation p-value)")
    ax.set_title("B. Cross-region match significance by arm", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Cross-region (V1 vs V2) niche matching depends on annotation method",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "2_cross_region_matching.png", bbox_inches="tight")
    plt.close(fig)


def fig3_annotation_axis(ndir: Path, perm_dir: Path, outdir: Path, k=8):
    combos = [("v1", "banksy"), ("v1", "stagate"), ("v2", "banksy"), ("v2", "stagate")]
    fig = plt.figure(figsize=(12, 8), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.55, wspace=0.35)

    # Panel A: 4 small multiples, per-domain composition cosine, colored by significance
    for i, (region, method) in enumerate(combos):
        ax = fig.add_subplot(gs[0, i])
        raw = pd.read_csv(ndir / f"{region}_{method}_annaxis_scvi_vs_scgpt.csv")
        pvals = pd.read_csv(perm_dir / f"{region}_{method}_annaxis_scvi_vs_scgpt_k8_pvalues.csv")
        merged = raw.merge(pvals[["domain", "p_value"]], on="domain")
        sig = merged["p_value"] < 0.05
        colors = np.where(sig, C_CRITICAL, INK_MUTED)
        x = np.arange(len(merged))
        ax.vlines(x, 0, merged["composition_cosine"], color=colors, linewidth=2)
        ax.scatter(x, merged["composition_cosine"], color=colors, s=48,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for xi, row in zip(x, merged.itertuples()):
            if row.p_value < 0.05:
                ax.annotate(f"{row.composition_cosine:.2f}", (xi, row.composition_cosine),
                            xytext=(0, -12), textcoords="offset points", ha="center",
                            fontsize=8, color=C_CRITICAL, fontweight="bold")
        style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(merged["domain"], fontsize=8)
        ax.set_ylim(0, 1.02)
        method_label = {"banksy": "Banksy", "stagate": "STAGATE"}
        ax.set_title(f"{region.upper()} x {method_label[method]}", fontsize=10,
                     fontweight="bold", color=INK)
        if i == 0:
            ax.set_ylabel("Composition cosine\n(scVI ~ scGPT)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CRITICAL,
               markeredgecolor=SURFACE, markersize=8, label="Significant divergence (p<0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_MUTED,
               markeredgecolor=SURFACE, markersize=8, label="Not significant"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=9)

    # Panel B: median cosine trend across k, hue=region, linestyle=method
    ax = fig.add_subplot(gs[1, :])
    kdf = pd.read_csv(ndir / "annotation_axis_cosine_by_k.csv")
    styles = {"banksy": "-", "stagate": "--"}
    colors = {"v1": C_V1, "v2": C_V2}
    for (region, method), sub in kdf.groupby(["region", "method"]):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["cosine_median"], color=colors[region],
                linestyle=styles[method], linewidth=2, marker="o", markersize=6,
                markerfacecolor=colors[region], markeredgecolor=SURFACE, markeredgewidth=1.5)
    style_axes(ax)
    ax.set_xlabel("Domain count (k)")
    ax.set_ylabel("Median composition cosine")
    ax.set_xticks(sorted(kdf["k"].unique()))
    ax.set_ylim(0, 1)
    ax.set_title("C. Q1 gap persists across the k-sweep (not a k=8 artifact)",
                 loc="left", fontsize=12, fontweight="bold", color=INK)
    method_label = {"banksy": "Banksy", "stagate": "STAGATE"}
    color_handles = [Line2D([0], [0], color=colors[r], lw=2, label=r.upper()) for r in ["v1", "v2"]]
    style_handles = [Line2D([0], [0], color=INK_SECONDARY, lw=2, linestyle=styles[m],
                             label=method_label[m]) for m in ["banksy", "stagate"]]
    leg1 = ax.legend(handles=color_handles, frameon=False, loc="lower left", title="Region")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, frameon=False, loc="lower right", title="Method")

    fig.suptitle("Annotation choice (scVI vs scGPT) shifts niche composition (Q1),\n"
                 "driven by specific domains", x=0.02, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=1.04)
    fig.savefig(outdir / "3_annotation_axis_composition.png", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Static figures for the 2x2 niche comparison")
    p.add_argument("--niche-comparison-dir", default="/scratch/cole.sy/neuro/results/niche_comparison")
    p.add_argument("--perm-dir", default="/scratch/cole.sy/neuro/results/niche_comparison_perm")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    ndir = Path(args.niche_comparison_dir)
    perm_dir = Path(args.perm_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1_cross_method_agreement(ndir, outdir)
    fig2_cross_region_matching(ndir, perm_dir, outdir, k=args.k)
    fig3_annotation_axis(ndir, perm_dir, outdir, k=args.k)
    print(f"wrote 3 figures to {outdir}")


if __name__ == "__main__":
    main()
