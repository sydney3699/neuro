#!/usr/bin/env python
"""
Combine the per-fold leave-genes-out CV results (cv_fold*.csv) into a single
report: ENVI vs kNN vs cell-type-mean held-out Spearman across all genes.

Outputs into <outdir>:
    cv_all.csv          every held-out gene with its 3 method correlations
    cv_summary.txt      median per method + paired win-rates (ENVI vs each)
    cv_comparison.png   per-gene distributions + ENVI-vs-baseline scatter
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="/scratch/cole.sy/neuro/results/envi_cv")
    args = p.parse_args()
    outdir = Path(args.outdir)

    files = sorted(glob.glob(str(outdir / "cv_fold*.csv")))
    if not files:
        raise SystemExit(f"no cv_fold*.csv in {outdir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(outdir / "cv_all.csv", index=False)

    # ENVI is only defined where the gene was actually imputed
    ev = df[df["imputed_present"] & df["envi"].notna()].copy()
    methods = ["envi", "knn", "celltype"]
    lines = [f"Leave-genes-out CV: {len(files)} folds, {len(df)} held-out genes "
             f"({ev['envi'].notna().sum()} with ENVI imputation)\n",
             "Median held-out Spearman:"]
    for m in methods:
        lines.append(f"  {m:9s} {df[m].median():.3f}")
    lines.append("\nENVI vs baselines (genes where ENVI defined):")
    for base in ["knn", "celltype"]:
        d = ev.dropna(subset=[base])
        wins = int((d["envi"] > d[base]).sum())
        lines.append(f"  ENVI > {base}: {wins}/{len(d)} genes "
                     f"(median delta {(d['envi']-d[base]).median():+.3f})")
    summary = "\n".join(lines)
    (outdir / "cv_summary.txt").write_text(summary + "\n")
    print(summary, flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    data = [df[m].dropna() for m in methods]
    ax1.boxplot(data, labels=[m.upper() for m in methods], showmeans=True)
    for i, d in enumerate(data, 1):
        ax1.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)),
                    d, s=8, alpha=0.4, color="#4477AA")
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set(ylabel="held-out Spearman", title="Per-gene held-out correlation by method")

    lim = [-0.2, 1.0]
    ax2.scatter(ev["knn"], ev["envi"], s=12, alpha=0.6, label="vs kNN", color="#EE6677")
    ax2.scatter(ev["celltype"], ev["envi"], s=12, alpha=0.6, label="vs cell-type", color="#228833")
    ax2.plot(lim, lim, "k--", lw=1)
    ax2.set(xlim=lim, ylim=lim, xlabel="baseline Spearman", ylabel="ENVI Spearman",
            title="ENVI vs baseline (above diagonal = ENVI wins)")
    ax2.legend()
    fig.tight_layout(); fig.savefig(outdir / "cv_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
