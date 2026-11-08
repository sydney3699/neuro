#!/usr/bin/env python
"""
How much of the Q1 "annotation method changes niche composition" result is seed noise?

WHY THIS EXISTS
---------------
The Q1 headline -- per-domain cell-type composition agrees between the scVI and scGPT
annotation arms at cosine median 0.65-0.94 (resolution-matched) -- was computed from
ONE scVI run and ONE scGPT run. seed_stability.py then showed the scGPT arm's label
transfer is broadly seed-fragile: across 10 Leiden seeds on a byte-identical latent
its cluster count moves 14-18, its reached-label count moves 10-13, and 10 different
reference labels flip in and out of recovery. A composition vector built on top of a
label set that unstable inherits that instability.

So the single-run Q1 cosine has an unknown error bar, and the question this script
answers is narrow: recompute the SAME statistic against each of the N scGPT seed
runs, holding the scVI arm and the domain partition fixed, and report the spread. If
the spread is small the Q1 number stands as published; if it is comparable to the
effect being claimed, the Q1 conclusion needs restating as a range.

WHY NO COMMOT h5ad IS READ
--------------------------
profile_niches.py computes composition alongside signaling and boundary stats, which
need the per-cell COMMOT substrate. Composition alone does not: it is exactly the
domain x label contingency table, row-normalised per domain. Recomputing just that
from the domain + annotation parquets makes the sweep ~40 cheap table joins instead
of ~40 loads of a multi-GB h5ad. Stage A validates this shortcut by reproducing the
PUBLISHED composition CSV for the reference arm before any seed is scored -- if that
does not match, nothing downstream is trustworthy and the script stops.

WHAT THE SPREAD DOES AND DOES NOT COVER
---------------------------------------
Only the annotation arm is reseeded. The domain partition, the COMMOT substrate, and
the scVI arm are all held fixed at their published values, so this isolates the
contribution of annotation-transfer seed noise to the Q1 statistic. It is NOT a full
pipeline error bar: domain-method seed noise (Banksy/STAGATE KMeans) and the scVI
arm's own instability are excluded by construction and would widen it further.

Outputs:
  q1_seed_cosine.csv      one row per (region, method, seed): median/min/mean cosine
  q1_seed_summary.csv     one row per (region, method): spread across seeds vs published
  q1_seed_labels.csv      per-seed label-set size and symmetric difference vs the scVI arm

Runs in the neuro env (pandas/numpy/scikit-learn).
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def composition(domains, labels):
    """Domain x label fraction table: the contingency table, row-normalised.

    `domains` and `labels` are per-cell Series already restricted to the same cells.
    Rows sum to 1, matching profile_niches.py's niche_composition output.
    """
    tab = pd.crosstab(domains, labels)
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def align(dfs):
    """Reindex to the union of columns, zero-filling -- same rule as compare_niches.py."""
    cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    return [d.reindex(columns=cols, fill_value=0.0) for d in dfs]


def main():
    p = argparse.ArgumentParser(
        description="Seed-noise error bar on the Q1 composition-cosine statistic")
    p.add_argument("--results-dir", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--k", type=int, default=8, help="domain count (domain_k{K} column)")
    p.add_argument("--ref-annotation",
                   default="annot_scvi/FB080_scvi_annotation.parquet",
                   help="the arm held FIXED, relative to --results-dir (the scVI arm)")
    p.add_argument("--ref-name", default="scvi")
    p.add_argument("--seed-glob", default="annot_seed_scgpt_leiden_S*/FB080_scgpt_annotation.parquet",
                   help="per-seed annotation parquets of the arm being reseeded")
    p.add_argument("--published-ann", default="scgpt",
                   help="annotation tag of the published run, for the niche_* dir name "
                        "holding the published cosine to compare the spread against")
    p.add_argument("--annotation-col", default="annotation")
    p.add_argument("--validate", action="store_true",
                   help="stage A: reproduce the published niche_composition.csv for the "
                        "reference arm and stop unless it matches")
    p.add_argument("--tol", type=float, default=1e-6, help="tolerance for the validation")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    from sklearn.metrics.pairwise import cosine_similarity

    RES = Path(args.results_dir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = [r for r in args.regions.split(",") if r]
    methods = [m for m in args.methods.split(",") if m]
    dom_col = f"domain_k{args.k}"

    ref_ann = pd.read_parquet(RES / args.ref_annotation)
    log(f"reference arm ({args.ref_name}): {RES / args.ref_annotation} {ref_ann.shape}")

    seed_paths = sorted(RES.glob(args.seed_glob))
    if not seed_paths:
        raise SystemExit(f"no seed annotation parquets matched {args.seed_glob}")

    def seed_of(path):
        m = re.search(r"_S(\d+)", str(path.parent.name))
        return int(m.group(1)) if m else None

    seeds = {seed_of(p): p for p in seed_paths if seed_of(p) is not None}
    log(f"{len(seeds)} seed runs: {sorted(seeds)}")

    # ---------- A. validate the composition shortcut against the published CSV ----------
    if args.validate:
        log("A. validating recomputed composition against the published niche_composition.csv")
        checked = 0
        for r in regions:
            for m in methods:
                pub_path = (RES / f"niche_{r}_{m}K{args.k}_{args.ref_name}" /
                            f"{r}_niche_composition.csv")
                dom_path = RES / m / f"{r}_{m}_domains.parquet"
                if not pub_path.exists() or not dom_path.exists():
                    log(f"   SKIP {r}x{m}: missing {pub_path.name if not pub_path.exists() else dom_path.name}")
                    continue
                dom = pd.read_parquet(dom_path)[dom_col]
                cells = dom.index.intersection(ref_ann.index)
                mine = composition(dom.loc[cells].astype(str),
                                   ref_ann.loc[cells, args.annotation_col].astype(str))
                pub = pd.read_csv(pub_path, index_col=0)
                mine.index = mine.index.astype(str)
                pub.index = pub.index.astype(str)
                common_r = mine.index.intersection(pub.index)
                a, b = align([mine.loc[common_r], pub.loc[common_r]])
                diff = float(np.abs(a.values - b.values).max())
                ok = diff <= args.tol
                log(f"   {r}x{m}: {len(common_r)} domains, max |diff| = {diff:.2e} "
                    f"-> {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    raise SystemExit(
                        f"composition shortcut does not reproduce {pub_path} "
                        f"(max abs diff {diff:.2e} > tol {args.tol}). Fix before trusting "
                        f"the seed spread.")
                checked += 1
        if not checked:
            raise SystemExit("validation requested but no (region, method) pair could be checked")
        log(f"   validated {checked} pair(s); shortcut reproduces the published tables")

    # ---------- B. per-seed Q1 cosine ----------
    log("B. recomputing the Q1 composition cosine against each seed")
    rows, label_rows = [], []
    for r in regions:
        for m in methods:
            dom_path = RES / m / f"{r}_{m}_domains.parquet"
            if not dom_path.exists():
                log(f"   SKIP {r}x{m}: no {dom_path}")
                continue
            dom = pd.read_parquet(dom_path)[dom_col]
            ref_cells = dom.index.intersection(ref_ann.index)
            ref_comp = composition(dom.loc[ref_cells].astype(str),
                                   ref_ann.loc[ref_cells, args.annotation_col].astype(str))
            ref_labels = set(ref_ann.loc[ref_cells, args.annotation_col].astype(str))

            # published single-run value, for the comparison the spread is about
            pub_path = (RES / f"niche_{r}_{m}K{args.k}_{args.published_ann}" /
                        f"{r}_niche_composition.csv")
            pub_median = np.nan
            if pub_path.exists():
                pub_comp = pd.read_csv(pub_path, index_col=0)
                pub_comp.index = pub_comp.index.astype(str)
                ci = ref_comp.index.intersection(pub_comp.index)
                A, B = align([ref_comp.loc[ci], pub_comp.loc[ci]])
                pub_median = float(np.median([
                    float(cosine_similarity(A.loc[[d]].values, B.loc[[d]].values)[0, 0])
                    for d in ci]))

            for s in sorted(seeds):
                ann = pd.read_parquet(seeds[s])
                cells = dom.index.intersection(ann.index).intersection(ref_ann.index)
                lab = ann.loc[cells, args.annotation_col].astype(str)
                comp_s = composition(dom.loc[cells].astype(str), lab)
                ci = ref_comp.index.intersection(comp_s.index)
                A, B = align([ref_comp.loc[ci], comp_s.loc[ci]])
                cos = [float(cosine_similarity(A.loc[[d]].values, B.loc[[d]].values)[0, 0])
                       for d in ci]
                rows.append({
                    "region": r, "method": m, "seed": s, "n_domains": len(ci),
                    "n_cells": len(cells),
                    "cosine_median": round(float(np.median(cos)), 4),
                    "cosine_min": round(float(np.min(cos)), 4),
                    "cosine_mean": round(float(np.mean(cos)), 4),
                    "published_cosine_median": (round(pub_median, 4)
                                                if np.isfinite(pub_median) else np.nan),
                })
                sl = set(lab.unique())
                label_rows.append({
                    "region": r, "method": m, "seed": s,
                    "n_labels_seed": len(sl), "n_labels_ref": len(ref_labels),
                    "n_shared": len(sl & ref_labels),
                    "only_in_seed": "; ".join(sorted(sl - ref_labels)),
                    "only_in_ref": "; ".join(sorted(ref_labels - sl)),
                })
            log(f"   {r}x{m}: {len(seeds)} seeds scored (published median "
                f"{pub_median:.4f})" if np.isfinite(pub_median) else
                f"   {r}x{m}: {len(seeds)} seeds scored (no published value found)")

    det = pd.DataFrame(rows)
    det.to_csv(outdir / "q1_seed_cosine.csv", index=False)
    pd.DataFrame(label_rows).to_csv(outdir / "q1_seed_labels.csv", index=False)
    log(f"   wrote q1_seed_cosine.csv ({len(det)} rows) + q1_seed_labels.csv")

    # ---------- C. spread vs the published single run ----------
    log("C. spread across seeds")
    srows = []
    for (r, m), g in det.groupby(["region", "method"]):
        pub = float(g["published_cosine_median"].iloc[0])
        srows.append({
            "region": r, "method": m, "n_seeds": len(g),
            "published_cosine_median": round(pub, 4) if np.isfinite(pub) else np.nan,
            "seed_mean": round(float(g["cosine_median"].mean()), 4),
            "seed_sd": round(float(g["cosine_median"].std()), 4),
            "seed_min": round(float(g["cosine_median"].min()), 4),
            "seed_max": round(float(g["cosine_median"].max()), 4),
            "seed_range": round(float(g["cosine_median"].max() - g["cosine_median"].min()), 4),
            # where the published run sits inside the seed distribution: if it is at
            # an extreme, the published figure was a lucky (or unlucky) draw
            "published_pctile_in_seeds": (
                round(float((g["cosine_median"] <= pub).mean() * 100), 1)
                if np.isfinite(pub) else np.nan),
        })
    summ = pd.DataFrame(srows)
    summ.to_csv(outdir / "q1_seed_summary.csv", index=False)
    log(f"   wrote q1_seed_summary.csv ({len(summ)} rows)")

    print(summ.to_string(index=False), flush=True)
    log("Reading it: seed_range small relative to the 0.65-0.94 spread the Q1 claim rests "
        "on => the published cosine stands. seed_range comparable to it => Q1 must be "
        "restated as a range, and a published figure at the 0th/100th percentile of the "
        "seed distribution was not a typical draw.")
    log("DONE.")


if __name__ == "__main__":
    main()
