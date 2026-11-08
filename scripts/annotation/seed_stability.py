#!/usr/bin/env python
"""
Is a label-recovery result a property of the pipeline, or of one lucky seed?

Motivation: re-training scVI with the SAME config and the SAME `--seed` produced a
different Leiden cluster count (36 vs the published 37) and a different reached-label
set (DID reach EN-IT-DL-2, which the published run missed). `scvi.settings.seed` does
not control GPU float reduction order, so a single run's reached-label set is not
reproducible even at fixed seed. Every single-run claim of the form "arm X never
reaches label Y" therefore needs a replicate check before it can be asserted.

This script consumes a SEED SWEEP -- the same annotate_cells.py invocation repeated
over N seeds -- and reports, per label, how often the transfer reached it. A label
reached at every seed is a stable property of the pipeline; a label reached at some
seeds and not others is run-to-run noise and must not be reported as a finding.

TWO FACTORS, REPORTED SEPARATELY
--------------------------------
The sweep is expected to carry two kinds of group:

  train  -- `--embedding scvi --seed S`: re-trains the latent per seed, so the
            variation includes BOTH scVI training stochasticity and Leiden.
  leiden -- `--embedding-parquet <fixed latent> --seed S`: the latent is held byte
            identical, so variation is Leiden/kNN stochasticity ALONE.

Comparing the two localizes the instability. If `leiden` is stable and `train` is
not, the latent is what moves; if `leiden` is also unstable, the clustering step is
the noise source regardless of embedding. scGPT's embedding is zero-shot inference
and deterministic, so only a `leiden` group is meaningful for it -- which is exactly
what makes it the control: any instability seen there is attributable to Leiden.

WHAT A STABILITY CLASS MEANS -- and what it does NOT
----------------------------------------------------
Classes are over the SEEDS SUPPLIED, nothing more. `stable_reached` with n_seeds=5
means 5/5, not a probabilistic guarantee; a label reached in 5/5 could still fail at
seed 6. With small N the honest reading of `stable_missed` is "no evidence of
recovery in N draws," and the binomial 95% upper bound on a 0/N result is reported
(`miss_upper95`) so the strength of that negative is explicit rather than implied.

As in label_recovery.py, "reached" remains a property of the embedding + Leiden +
centroid-transfer pipeline, NOT a statement about information content in the latent.
A label can be stably missed by the transfer and still be near-perfectly decodable
per-cell -- see embedding_discriminability.py, which is the test for that question.

Inputs: per-seed run directories named `<group>_S<seed>` holding the usual
`{tag}_{emb}_cluster_map.csv` (and optionally `{tag}_{emb}_annotation.parquet`).

Outputs:
  seed_label_stability.csv  one row per (group, label): reach fraction + class
  seed_run_summary.csv      one row per (group, seed): cluster count, labels reached
  seed_cluster_counts.csv   per-group cluster-count dispersion across seeds
  seed_verdict.csv          per (group, focus label) verdict, for the brief

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


def parse_seed_run(dirname):
    """('<group>', seed) from a run directory named '<group>_S<seed>'.

    The seed suffix is required: without it the run cannot be placed in a sweep, and
    silently folding it in as another replicate of an unknown group would corrupt
    every stability fraction downstream.
    """
    m = re.search(r"^(.*)_S(\d+)$", dirname)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def binom_upper95(n):
    """Upper 95% bound on p given 0 successes in n trials: 1 - 0.05**(1/n).

    The rule-of-three approximation (3/n) is close for larger n, but N here is small
    enough (5-10) that the exact form is worth using.
    """
    return float(1.0 - 0.05 ** (1.0 / n)) if n > 0 else np.nan


def load_sweep(results_dir, glob):
    """One DataFrame per per-seed cluster_map CSV, concatenated, with group/seed."""
    paths = sorted(Path(results_dir).glob(glob))
    if not paths:
        raise SystemExit(f"no cluster_map CSVs found under {results_dir}/{glob}")
    frames, skipped = [], []
    for path in paths:
        group, seed = parse_seed_run(path.parent.name)
        if group is None:
            skipped.append(path.parent.name)
            continue
        df = pd.read_csv(path)
        if "leiden" not in df.columns:
            df = df.rename(columns={df.columns[0]: "leiden"})
        if "annotation" not in df.columns:
            log(f"  SKIP {path.parent.name}: no 'annotation' column")
            continue
        # The factor under test is encoded in the group name by the sbatch that wrote
        # it; fall back to 'unknown' rather than guessing from the directory contents.
        factor = ("train" if "_train" in group else
                  "leiden" if "_leiden" in group else "unknown")
        emb = next((c for c in ("scgpt", "scvi", "pca", "uce") if c in group), "unknown")
        df = df.assign(group=group, seed=seed, factor=factor, embedding=emb)
        frames.append(df)
        log(f"  {path.parent.name}: {len(df)} clusters, "
            f"{df['annotation'].nunique()} distinct labels")
    if skipped:
        log(f"  WARNING: {len(skipped)} dirs lack a _S<seed> suffix and were ignored: "
            f"{', '.join(sorted(skipped)[:5])}")
    if not frames:
        raise SystemExit("no usable per-seed runs parsed")
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(
        description="Seed-stability of cluster->centroid label recovery")
    p.add_argument("--results-dir", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--glob", default="annot_seed_*/*_cluster_map.csv",
                   help="glob under --results-dir; parent dirs must end in _S<seed>")
    p.add_argument("--focus-prefix", default="EN-IT",
                   help="label-family prefix for the verdict table; '' disables")
    p.add_argument("--focus-labels", default="",
                   help="explicit comma-separated focus labels, overriding --focus-prefix")
    p.add_argument("--all-labels", default="",
                   help="comma-separated full reference label universe. Without it the "
                        "universe is the union of labels some seed reached, so a label "
                        "NO seed ever reached is invisible (and cannot be scored as "
                        "stably missed). Strongly recommended for a real verdict.")
    p.add_argument("--all-labels-csv", default="",
                   help="alternative to --all-labels: a reference-centroid CSV whose "
                        "index column is the label universe (ref_centroids_snrna.csv)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"A. loading seed sweep from {args.results_dir}/{args.glob}")
    cm = load_sweep(args.results_dir, args.glob)
    has_runnerup = {"second_label", "second_corr"}.issubset(cm.columns)
    if not has_runnerup:
        log("  WARNING: no second_label column -- runner-up columns omitted")
    groups = sorted(cm["group"].unique())
    log(f"  {len(groups)} group(s): " +
        "; ".join(f"{g} (n_seeds={cm[cm['group'] == g]['seed'].nunique()})" for g in groups))

    # --- label universe ---------------------------------------------------------
    # Prefer an explicit universe: restricting to labels some seed happened to reach
    # makes a never-reached label unscoreable, which is precisely the case the
    # sweep exists to adjudicate.
    if args.all_labels_csv:
        universe = sorted(map(str, pd.read_csv(args.all_labels_csv, index_col=0).index))
        log(f"B. label universe: {len(universe)} labels from {args.all_labels_csv}")
    elif args.all_labels:
        universe = sorted(set(args.all_labels.split(",")))
        log(f"B. label universe: {len(universe)} labels from --all-labels")
    else:
        u = set(cm["annotation"].dropna().astype(str).unique())
        if has_runnerup:
            u |= set(cm["second_label"].dropna().astype(str).unique())
        universe = sorted(u)
        log(f"B. label universe: {len(universe)} labels (union over seeds; labels no "
            f"seed reached are NOT represented -- pass --all-labels-csv for a real verdict)")
    if args.focus_labels:
        focus = [c for c in args.focus_labels.split(",") if c in universe]
    elif args.focus_prefix:
        focus = [c for c in universe if str(c).startswith(args.focus_prefix)]
    else:
        focus = list(universe)
    log(f"   focus labels ({len(focus)}): {', '.join(map(str, focus))}")

    # --- per-(group, seed) run summary -------------------------------------------
    log("C. per-(group, seed) run summary")
    run_rows = []
    for (group, seed), g in cm.groupby(["group", "seed"]):
        reached = set(g["annotation"].dropna().astype(str))
        run_rows.append({
            "group": group, "seed": seed,
            "factor": g["factor"].iloc[0], "embedding": g["embedding"].iloc[0],
            "n_clusters": len(g),
            "n_labels_reached": len(reached),
            "n_focus_reached": len([f for f in focus if f in reached]),
            "focus_missed": "; ".join(f for f in focus if f not in reached),
            "median_best_corr": (round(float(g["best_corr"].median()), 4)
                                 if "best_corr" in g.columns else np.nan),
        })
    runs = pd.DataFrame(run_rows).sort_values(["group", "seed"])
    runs.to_csv(outdir / "seed_run_summary.csv", index=False)
    log(f"   wrote seed_run_summary.csv ({len(runs)} runs)")

    # --- cluster-count dispersion -------------------------------------------------
    # The published-vs-rerun discrepancy first showed up as a cluster-count change
    # (37 -> 36), so dispersion here is the cheapest summary of run-to-run drift.
    cnt = (runs.groupby(["group", "factor", "embedding"])["n_clusters"]
           .agg(n_seeds="count", mean="mean", sd="std", min="min", max="max")
           .reset_index())
    cnt["range"] = cnt["max"] - cnt["min"]
    lab_cnt = (runs.groupby("group")["n_labels_reached"]
               .agg(labels_mean="mean", labels_sd="std",
                    labels_min="min", labels_max="max").reset_index())
    cnt = cnt.merge(lab_cnt, on="group", how="left")
    cnt.to_csv(outdir / "seed_cluster_counts.csv", index=False)
    log(f"   wrote seed_cluster_counts.csv")

    # --- per-(group, label) stability ---------------------------------------------
    log("D. per-(group, label) reach fraction across seeds")
    rows = []
    for group, g in cm.groupby("group"):
        seeds = sorted(g["seed"].unique())
        n_seeds = len(seeds)
        # reached-by-seed sets, so a label assigned by 2 clusters in one seed still
        # counts once for that seed (we are scoring recovery, not cluster share)
        by_seed = {s: set(gs["annotation"].dropna().astype(str))
                   for s, gs in g.groupby("seed")}
        ru_by_seed = ({s: set(gs["second_label"].dropna().astype(str))
                       for s, gs in g.groupby("seed")} if has_runnerup else {})
        prim_counts = (g.groupby(["seed", "annotation"]).size()
                       .rename("n").reset_index())
        for label in universe:
            hits = [s for s in seeds if label in by_seed.get(s, ())]
            frac = len(hits) / n_seeds
            sub = prim_counts[prim_counts["annotation"] == label]
            row = {
                "group": group,
                "factor": g["factor"].iloc[0], "embedding": g["embedding"].iloc[0],
                "label": label, "n_seeds": n_seeds, "n_seeds_reached": len(hits),
                "reach_frac": round(frac, 4),
                "stability": ("stable_reached" if frac == 1.0 else
                              "stable_missed" if frac == 0.0 else "unstable"),
                "seeds_reached": ";".join(map(str, hits)),
                "mean_clusters_when_reached": (round(float(sub["n"].mean()), 3)
                                               if len(sub) else 0.0),
                # strength of a 0/N negative; NaN unless the label was never reached
                "miss_upper95": round(binom_upper95(n_seeds), 4) if frac == 0.0 else np.nan,
            }
            if has_runnerup:
                ru_hits = [s for s in seeds if label in ru_by_seed.get(s, ())]
                row["n_seeds_runnerup"] = len(ru_hits)
                row["near_miss_only"] = bool(frac == 0.0 and len(ru_hits) > 0)
            rows.append(row)
    stab = pd.DataFrame(rows).sort_values(["group", "stability", "label"])
    stab.to_csv(outdir / "seed_label_stability.csv", index=False)
    log(f"   wrote seed_label_stability.csv ({len(stab)} rows)")

    # --- verdict table on the focus labels ----------------------------------------
    log("E. focus-label verdict")
    ver = stab[stab["label"].isin(focus)].copy()
    ver["citable"] = ver["stability"] != "unstable"
    ver["claim"] = np.where(
        ver["stability"] == "stable_reached", "reached at every seed",
        np.where(ver["stability"] == "stable_missed",
                 "never reached (0/N; see miss_upper95)",
                 "SEED-DEPENDENT -- do not cite as a finding"))
    ver[["group", "factor", "embedding", "label", "n_seeds", "n_seeds_reached",
         "reach_frac", "stability", "citable", "claim", "miss_upper95"]
        + (["n_seeds_runnerup", "near_miss_only"] if has_runnerup else [])
        ].to_csv(outdir / "seed_verdict.csv", index=False)
    log(f"   wrote seed_verdict.csv ({len(ver)} rows)")

    # --- console view --------------------------------------------------------------
    log("SUMMARY: cluster-count dispersion across seeds")
    print(cnt.to_string(index=False), flush=True)

    log("SUMMARY: focus-label recovery by seed")
    piv = (stab[stab["label"].isin(focus)]
           .pivot_table(index="group", columns="label", values="reach_frac"))
    print(piv.to_string(), flush=True)

    unstable = stab[stab["stability"] == "unstable"]
    log(f"UNSTABLE labels (reached at some seeds, missed at others): {len(unstable)}")
    for r in unstable.itertuples():
        log(f"  {r.group}: {r.label} -- {r.n_seeds_reached}/{r.n_seeds} seeds "
            f"(seeds {r.seeds_reached})")
    if unstable.empty:
        log("  none -- every label's recovery was consistent across the seeds tested")
    log("NOTE: classes describe only the seeds supplied. 'stable_missed' means no "
        "recovery in N draws, not impossibility; read miss_upper95 alongside it. "
        "And 'missed' remains a transfer-level result -- see "
        "embedding_discriminability.py for whether the latent separates those cells.")
    log("DONE.")


if __name__ == "__main__":
    main()
