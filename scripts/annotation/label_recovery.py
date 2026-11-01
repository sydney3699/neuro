#!/usr/bin/env python
"""
Which reference cell-type labels does each embedding's LABEL TRANSFER actually
reach, and how does that change with Leiden resolution?

Motivation: the scVI arm assigns clusters to all four fine EN-IT subtypes, while
the scGPT arm assigns none to EN-IT-L4-V1 or EN-IT-DL-1 -- and keeps assigning
none even when its resolution is raised until its cluster count matches (R3.5:
36 clusters vs scVI's 37) or exceeds (R4.0: 42) scVI's. That rules out "scGPT
just made fewer clusters" as the explanation, which is the one confound the
cluster-count logs could already address. This script persists that comparison
as a reproducible artifact instead of a hand-made cross-tab.

WHAT THIS MEASURES -- and what it does NOT
------------------------------------------
The unit here is a CLUSTER-to-REFERENCE-CENTROID assignment, i.e. the output of
annotate_cells.py's label transfer (Leiden -> mean expression -> argmax Pearson
correlation against the snRNA reference centroids). A label is "reached" when some
cluster's centroid correlates best with that label's centroid.

So a label with zero primary assignments means: *this embedding + Leiden
resolution + centroid-transfer pipeline never produced a cluster whose centroid
matched that reference type best.* It does NOT establish that the embedding lacks
the information to separate those cells. A cluster can contain EN-IT-L4-V1 cells
and still have its centroid land on EN-IT-UL-1 -- clustering granularity, cluster
impurity, and the argmax step can each erase a distinction the latent space
actually encodes. Settling the information-content question needs a per-cell test
(hold out labels, ask whether a classifier/kNN can recover them from the embedding
directly, no Leiden step) -- not this table.

The runner-up columns are the hedge against over-reading the primary count:
`n_clusters_runnerup` / `best_second_corr` use the retained second_label /
second_corr, so a label that is never the argmax but is persistently the close
runner-up reads as "ranked but not selected" rather than "absent." Treat a label
with 0 primary and 0 runner-up assignments as a stronger (still transfer-level)
negative than one with 0 primary but several near-miss runner-ups.

Inputs (no embedding/Leiden rerun -- reads committed run outputs only):
  --results-dir + --glob   the per-cluster {tag}_{emb}_cluster_map.csv files
                           (full schema: annotation, second_label, second_corr, ...)
  --detail-csv             fallback: confidence_recalibration_detail.csv, which
                           carries run/leiden/annotation but NOT second_label, so
                           runner-up columns are omitted in that mode
  --annotation-glob        optional {tag}_{emb}_annotation.parquet files, to add
                           per-cell counts (cluster counts alone over-weight small
                           clusters; a label held by 1 huge cluster vs 3 tiny ones
                           is not the same thing)

Outputs:
  label_recovery_long.csv     one row per (run, label)
  label_recovery_matrix.csv   runs x focus labels, primary cluster counts (the
                              resolution-grid table)
  label_recovery_summary.csv  one row per run: cluster count, labels reached,
                              focus labels reached/missed

Runs in the neuro env (pandas/numpy only).
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_run(run):
    """(embedding, resolution, arm) from a run name like 'annot_scgpt_resgrid_R3.5/...'.

    Resolution is only recoverable for the resgrid runs, which encode it in the
    name; the base runs were all produced at the annotate_cells.py default, so
    they are tagged with --base-resolution rather than guessed per run.
    """
    stem = run.split("/")[0]
    # The embedding name is usually in the directory (annot_scgpt/...), but some runs
    # only carry it in the filename (annot_test/FB080_pca_cluster_map), so check both.
    emb = "unknown"
    for cand in ("scgpt", "scvi", "pca", "uce"):
        if cand in run:
            emb = cand
            break
    m = re.search(r"resgrid_R([0-9.]+)", stem)
    res = float(m.group(1)) if m else np.nan
    if "coarse" in stem:
        arm = "coarse"
    elif m:
        arm = "resgrid"
    else:
        arm = "fine"
    return emb, res, arm


def load_cluster_maps(results_dir, glob):
    """Full-schema path: one DataFrame per cluster_map CSV, concatenated."""
    paths = sorted(Path(results_dir).glob(glob))
    if not paths:
        raise SystemExit(f"no cluster_map CSVs found under {results_dir}/{glob}")
    frames = []
    for path in paths:
        run = path.parent.name + "/" + path.stem
        df = pd.read_csv(path)
        if "leiden" not in df.columns:
            df = df.rename(columns={df.columns[0]: "leiden"})
        if "annotation" not in df.columns:
            log(f"  SKIP {run}: no 'annotation' column")
            continue
        df["run"] = run
        frames.append(df)
        log(f"  {run}: {len(df)} clusters, {df['annotation'].nunique()} distinct labels")
    return pd.concat(frames, ignore_index=True)


def load_detail(detail_csv):
    """Fallback path: the recalibration detail CSV (no second_label/second_corr)."""
    df = pd.read_csv(detail_csv)
    need = {"run", "leiden", "annotation"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{detail_csv} missing columns: {missing}")
    log(f"  {detail_csv}: {len(df)} cluster rows across {df['run'].nunique()} runs")
    return df


def load_cell_counts(annotation_glob, results_dir):
    """Per-(run, label) cell counts from the per-cell annotation parquets, if given."""
    paths = sorted(Path(results_dir).glob(annotation_glob))
    if not paths:
        log(f"  no annotation parquets matched {annotation_glob}; skipping cell counts")
        return None
    rows = []
    for path in paths:
        run = path.parent.name + "/" + path.stem.replace("_annotation", "_cluster_map")
        per_cell = pd.read_parquet(path, columns=["annotation"])
        vc = per_cell["annotation"].value_counts()
        n_total = int(vc.sum())
        for label, n in vc.items():
            rows.append({"run": run, "label": label, "n_cells": int(n),
                         "cell_frac": round(float(n) / n_total, 4)})
        log(f"  {run}: {n_total} cells over {len(vc)} labels")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="Reference-label recovery of cluster->centroid label transfer, per run")
    p.add_argument("--results-dir", default="/scratch/cole.sy/neuro/results",
                   help="dir holding annot_*/ subdirs with *_cluster_map.csv")
    p.add_argument("--glob", default="annot_*/*_cluster_map.csv",
                   help="glob under --results-dir for the cluster_map CSVs")
    p.add_argument("--detail-csv", default="",
                   help="fallback input: confidence_recalibration_detail.csv "
                        "(used instead of --glob; omits the runner-up columns)")
    p.add_argument("--annotation-glob", default="",
                   help="optional glob for per-cell *_annotation.parquet, to add cell counts "
                        "(e.g. 'annot_*/*_annotation.parquet')")
    p.add_argument("--focus-prefix", default="EN-IT",
                   help="label-family prefix for the matrix output; '' disables prefix selection")
    p.add_argument("--focus-labels", default="",
                   help="explicit comma-separated focus labels, overriding --focus-prefix")
    p.add_argument("--all-labels", default="",
                   help="optional comma-separated full reference label universe. Without it the "
                        "universe is the union of labels any run reached, so a label NO arm ever "
                        "reached is invisible to this table.")
    p.add_argument("--base-resolution", type=float, default=1.0,
                   help="Leiden resolution of the non-resgrid runs (annotate_cells.py default)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.detail_csv:
        log(f"A. loading cluster assignments (fallback detail mode)")
        cm = load_detail(args.detail_csv)
        has_runnerup = False
    else:
        log(f"A. loading cluster_map CSVs from {args.results_dir}/{args.glob}")
        cm = load_cluster_maps(args.results_dir, args.glob)
        has_runnerup = {"second_label", "second_corr"}.issubset(cm.columns)
        if not has_runnerup:
            log("  WARNING: no second_label/second_corr columns -- runner-up columns omitted")

    cm[["embedding", "resolution", "arm"]] = cm["run"].apply(
        lambda r: pd.Series(parse_run(r), index=["embedding", "resolution", "arm"]))
    cm["resolution"] = cm["resolution"].fillna(args.base_resolution)
    # Coarse runs transfer against the 16-class coarse reference, fine runs against
    # the 35 fine subtypes -- different vocabularies. Pooling them would score every
    # coarse run as "missing" every fine label (and vice versa) purely by construction,
    # so recovery is always computed WITHIN a namespace.
    cm["namespace"] = np.where(cm["arm"] == "coarse", "coarse", "fine")

    # label universe, per namespace: explicit if given, else everything some run in
    # that namespace reached (primary or runner-up). Narrower than the true reference
    # taxonomy -- see --all-labels.
    universe, focus = {}, {}
    for ns, g in cm.groupby("namespace"):
        if args.all_labels:
            u = sorted(set(args.all_labels.split(",")))
            log(f"B[{ns}]. label universe: {len(u)} labels from --all-labels")
        else:
            u = set(g["annotation"].dropna().unique())
            if has_runnerup:
                u |= set(g["second_label"].dropna().unique())
            u = sorted(u)
            log(f"B[{ns}]. label universe: {len(u)} labels (union over {ns} runs; "
                f"labels no {ns} arm reached are NOT represented)")
        universe[ns] = u
        if args.focus_labels:
            f = [c for c in args.focus_labels.split(",") if c in u]
        elif args.focus_prefix:
            f = [c for c in u if str(c).startswith(args.focus_prefix)]
        else:
            f = list(u)
        focus[ns] = f
        log(f"   [{ns}] focus labels ({len(f)}): {', '.join(map(str, f))}")

    cells = load_cell_counts(args.annotation_glob, args.results_dir) if args.annotation_glob else None

    # ---------- long table: one row per (run, label) ----------
    log("C. tabulating per-(run, label) recovery")
    long_rows = []
    for run, g in cm.groupby("run"):
        emb, res, arm = g["embedding"].iloc[0], g["resolution"].iloc[0], g["arm"].iloc[0]
        ns = g["namespace"].iloc[0]
        n_clusters = len(g)
        primary = g["annotation"].value_counts()
        if has_runnerup:
            runnerup = g["second_label"].value_counts()
            # strongest near-miss: best second_corr among clusters ranking this label 2nd
            best_second = g.groupby("second_label")["second_corr"].max()
        for label in universe[ns]:
            n_prim = int(primary.get(label, 0))
            row = {
                "run": run, "embedding": emb, "resolution": res, "arm": arm,
                "namespace": ns,
                "n_clusters_in_run": n_clusters, "label": label,
                "n_clusters_assigned": n_prim,
                "reached_by_transfer": bool(n_prim > 0),
            }
            if has_runnerup:
                n_run = int(runnerup.get(label, 0))
                row["n_clusters_runnerup"] = n_run
                row["best_second_corr"] = (round(float(best_second.get(label, np.nan)), 4)
                                           if n_run else np.nan)
                # 0 primary but >=1 runner-up = ranked-but-not-selected, a weaker
                # negative than never appearing in either slot.
                row["near_miss_only"] = bool(n_prim == 0 and n_run > 0)
            if cells is not None:
                sub = cells[(cells["run"] == run) & (cells["label"] == label)]
                row["n_cells"] = int(sub["n_cells"].iloc[0]) if len(sub) else 0
                row["cell_frac"] = float(sub["cell_frac"].iloc[0]) if len(sub) else 0.0
            long_rows.append(row)
    long = pd.DataFrame(long_rows).sort_values(["namespace", "embedding", "arm", "resolution", "label"])
    long.to_csv(outdir / "label_recovery_long.csv", index=False)
    log(f"   wrote label_recovery_long.csv ({len(long)} rows)")

    # ---------- matrix: runs x focus labels (the resolution-grid table) ----------
    # One matrix per namespace; stacked with a namespace column. Blank (not 0) for a
    # label outside a run's namespace, so an unreachable-by-construction label never
    # reads as a zero-recovery result.
    log("D. building run x focus-label matrix (per namespace)")
    mats = []
    for ns, g in long.groupby("namespace"):
        fl = g[g["label"].isin(focus[ns])]
        if fl.empty:
            continue
        m = fl.pivot_table(index=["namespace", "embedding", "arm", "resolution", "run",
                                  "n_clusters_in_run"],
                           columns="label", values="n_clusters_assigned",
                           aggfunc="sum", fill_value=0).reset_index()
        mats.append(m.sort_values(["embedding", "arm", "resolution"]))
    mat = pd.concat(mats, ignore_index=True) if mats else pd.DataFrame()
    mat.to_csv(outdir / "label_recovery_matrix.csv", index=False)
    log(f"   wrote label_recovery_matrix.csv ({len(mat)} runs; "
        f"{ {ns: len(f) for ns, f in focus.items()} } focus labels per namespace)")

    # ---------- per-run summary ----------
    log("E. per-run summary")
    sum_rows = []
    for run, g in long.groupby("run"):
        ns = g["namespace"].iloc[0]
        gf = g[g["label"].isin(focus[ns])]
        reached = gf[gf["n_clusters_assigned"] > 0]["label"].tolist()
        missed = gf[gf["n_clusters_assigned"] == 0]["label"].tolist()
        row = {
            "run": run, "embedding": g["embedding"].iloc[0], "arm": g["arm"].iloc[0],
            "namespace": ns, "resolution": g["resolution"].iloc[0],
            "n_clusters_in_run": int(g["n_clusters_in_run"].iloc[0]),
            "n_labels_reached": int((g["n_clusters_assigned"] > 0).sum()),
            "n_focus_reached": len(reached), "n_focus_total": len(focus[ns]),
            "focus_missed": "; ".join(map(str, missed)),
        }
        if has_runnerup:
            gm = gf[gf["near_miss_only"]]
            row["n_focus_near_miss_only"] = int(len(gm))
            row["focus_near_miss_only"] = "; ".join(
                f"{r.label}(r2={r.best_second_corr})" for r in gm.itertuples())
        sum_rows.append(row)
    summ = pd.DataFrame(sum_rows).sort_values(["namespace", "embedding", "arm", "resolution"])
    summ.to_csv(outdir / "label_recovery_summary.csv", index=False)
    log(f"   wrote label_recovery_summary.csv ({len(summ)} runs)")

    # ---------- console view ----------
    for ns in sorted(focus):
        sub = mat[mat["namespace"] == ns] if "namespace" in mat.columns else mat
        if sub.empty:
            continue
        log(f"SUMMARY [{ns} labels] (clusters assigned to each focus label, by run)")
        cols = ["embedding", "arm", "resolution", "n_clusters_in_run"] + focus[ns]
        print(sub[[c for c in cols if c in sub.columns]].to_string(index=False), flush=True)
    log("Focus labels never reached by label transfer (within each run's namespace):")
    for r in summ.itertuples():
        if r.focus_missed:
            extra = ""
            if has_runnerup and getattr(r, "n_focus_near_miss_only", 0):
                extra = f"  [runner-up only: {r.focus_near_miss_only}]"
            log(f"  {r.run} (k={r.n_clusters_in_run} clusters): {r.focus_missed}{extra}")
    log("NOTE: 'never reached' is a property of this embedding+resolution+centroid-transfer "
        "pipeline, NOT evidence that the embedding cannot separate those cells. A per-cell "
        "held-out-label test is required for that claim.")
    log("DONE.")


if __name__ == "__main__":
    main()
