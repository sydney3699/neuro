#!/usr/bin/env python
"""
Recalibrate the annotate_cells.py low-confidence flag on EXISTING runs, with no
embedding/Leiden rerun. Reads the per-cluster `{tag}_{emb}_cluster_map.csv` files
(which already retain best_corr / margin / metric_consensus / low_confidence) and
re-applies the new adaptive flag (percentile margin + 3-metric consensus) from
annotate_cells.compute_low_confidence, reporting old-vs-new flag rates so a
sensible --margin-percentile can be chosen before touching any pipeline run.

The old rule `margin < 0.05` flagged 59-84% of clusters (driven entirely by the
fixed margin cut). This quantifies how far the new rule pulls that back and which
specific clusters change, for QC review.

Runs in the neuro env (pandas/numpy only). Import-safe: annotate_cells' heavy deps
(scanpy/scvi) are all lazy, so importing compute_low_confidence is cheap.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neurospatial.confidence import compute_low_confidence


def main():
    p = argparse.ArgumentParser(description="Recalibrate low-confidence flags on existing cluster_map CSVs")
    p.add_argument("--results-dir", default="/scratch/cole.sy/neuro/results",
                   help="dir holding annot_*/ subdirs with *_cluster_map.csv")
    p.add_argument("--glob", default="annot_*/*_cluster_map.csv",
                   help="glob under --results-dir for the cluster_map CSVs")
    p.add_argument("--margin-percentile", type=float, default=25.0)
    p.add_argument("--min-corr", type=float, default=0.3)
    p.add_argument("--min-margin-floor", type=float, default=None,
                   help="optional absolute margin floor (old behavior); disabled by default")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    RES = Path(args.results_dir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(RES.glob(args.glob))
    if not paths:
        raise SystemExit(f"no cluster_map CSVs found under {RES}/{args.glob}")

    summary_rows, detail_rows = [], []
    for path in paths:
        run = path.parent.name + "/" + path.stem  # e.g. annot_scvi/FB080_scvi_cluster_map
        df = pd.read_csv(path)
        # tolerate either an indexed 'leiden' column or the first column being it
        if "leiden" not in df.columns:
            df = df.rename(columns={df.columns[0]: "leiden"})
        need = {"best_corr", "margin", "metric_consensus"}
        if not need.issubset(df.columns):
            print(f"  SKIP {run}: missing {need - set(df.columns)}", flush=True)
            continue
        old_low = df["low_confidence"].astype(bool).values if "low_confidence" in df.columns else np.zeros(len(df), bool)
        new_low = compute_low_confidence(
            df["best_corr"].values, df["margin"].values, df["metric_consensus"].values,
            min_corr=args.min_corr, margin_percentile=args.margin_percentile,
            min_margin_floor=args.min_margin_floor)
        n = len(df)
        summary_rows.append({
            "run": run, "n_clusters": n,
            "old_flagged": int(old_low.sum()), "old_rate": round(float(old_low.mean()), 3),
            "new_flagged": int(new_low.sum()), "new_rate": round(float(new_low.mean()), 3),
            "margin_p_thr": round(float(np.percentile(df["margin"].values, args.margin_percentile)), 4),
            "n_full_consensus": int((df["metric_consensus"] == 3).sum()),
        })
        for _, row, ol, nl in zip(range(n), df.itertuples(index=False), old_low, new_low):
            detail_rows.append({
                "run": run, "leiden": row.leiden, "annotation": row.annotation,
                "best_corr": row.best_corr, "margin": row.margin,
                "metric_consensus": row.metric_consensus,
                "old_low": bool(ol), "new_low": bool(nl), "changed": bool(ol) != bool(nl),
            })
        print(f"  {run}: {n} clusters | old {int(old_low.sum())}/{n} ({old_low.mean():.0%}) "
              f"-> new {int(new_low.sum())}/{n} ({new_low.mean():.0%})", flush=True)

    summ = pd.DataFrame(summary_rows)
    det = pd.DataFrame(detail_rows)
    summ.to_csv(outdir / "confidence_recalibration_summary.csv", index=False)
    det.to_csv(outdir / "confidence_recalibration_detail.csv", index=False)
    print(f"\nwrote confidence_recalibration_summary.csv ({len(summ)} runs) + detail "
          f"({len(det)} clusters, {int(det['changed'].sum())} changed) to {outdir}", flush=True)


if __name__ == "__main__":
    main()
