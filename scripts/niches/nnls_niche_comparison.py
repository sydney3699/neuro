#!/usr/bin/env python
"""
Regenerate the niche comparison artifacts on top of the NNLS label transfer.

WHY THE OLD ARTIFACTS NEEDED REPLACING
--------------------------------------
The published `niche_comparison/` CSVs report the Q1 annotation-axis statistic
(per-domain composition cosine, scVI vs scGPT) as single-run point estimates.
q1_seed_variance.py showed those are not reproducible to the precision reported:
re-seeding only the scGPT arm, with domains and the scVI arm held fixed, moves the
statistic by 0.203-0.361 -- as large as the effect being claimed -- and the published
v1xbanksy figure sits at the 10th percentile of its own seed distribution.

WHY COMPOSITION MUST CONSUME THE MIXTURE, NOT THE HARD LABELS
--------------------------------------------------------------
transfer_fix.py's NNLS mode writes a per-cell annotation parquet whose label is the
ARGMAX of the cluster's mixture, purely so downstream code that expects one label per
cell keeps working. For the scGPT arm that argmax never lands on EN-IT-L4-V1, DL-1 or
UL-1 (is_argmax_somewhere = False, mean max weight 0.13-0.19), so a composition built
from those hard labels recovers NOTHING that the baseline transfer missed -- the whole
gain of NNLS lives in the weights. This script therefore builds composition as the
mean MIXTURE VECTOR over the cells of a domain:

    composition[domain, type] = mean over cells c in domain of mixture[cluster(c), type]

i.e. the expected cell-type fraction under the deconvolution. Stage B reports the
hard-label composition alongside it so the difference is explicit rather than assumed.

WHY THE RAW COSINE IS NOT COMPARABLE TO THE PUBLISHED NUMBER
-------------------------------------------------------------
Under NNLS both arms produce DENSE composition vectors spanning all 35 reference
types. The baseline produced SPARSE vectors over each arm's reached labels only
(scVI 16-17, scGPT 10-13), aligned by zero-filled union -- so the two arms' vectors
had partly disjoint support. Cosine between two dense vectors that both carry mass
everywhere is systematically higher than between two sparse vectors with disjoint
support, REGARDLESS of whether the arms agree any better. Quoting an NNLS cosine
against the published cosine would therefore manufacture an improvement.

So the headline statistic here is representation-invariant: the MATCHED-vs-MISMATCHED
GAP. For a given (region, method, seed pair):

    matched   = median over domains d of cos(scvi_comp[d], scgpt_comp[d])
    mismatched= mean over d != d' of cos(scvi_comp[d], scgpt_comp[d'])
    gap       = matched - mismatched

`matched` alone answers "do the arms agree"; it is inflated by density. `gap` answers
"do the arms agree on what makes each domain DIFFERENT from the others", and a
uniform density shift cancels out of it. A gap near zero means the arms' apparent
agreement carries no domain-specific information -- every domain looks like every
other -- however high the raw cosine is. Both are reported; the gap is the one to
cite, and it is computed identically for the baseline hard-label arm so the two
representations can actually be compared.

Spread: every statistic is computed over all (scvi_seed x scgpt_seed) pairs, so the
output carries the seed error bar the published artifacts lacked.

Outputs (per --outdir):
  nnls_annotation_axis.csv      per (region, method, seed pair): matched/mismatched/gap
  nnls_annotation_axis_summary.csv  per (region, method): spread over seed pairs,
                                    with the baseline hard-label comparison
  nnls_composition_{region}_{method}_{arm}_S{seed}.csv   the weighted compositions
  nnls_label_mass.csv           per arm: total composition mass on each label, hard
                                vs weighted -- the audit of what the mixture recovers
  nnls_crossregion_match.csv    V1-vs-V2 mutual-best niche matching per arm/seed

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


def weighted_composition(domains, leiden, mixture):
    """Domain x type table of mean mixture weight.

    domains/leiden are per-cell Series on the same index; mixture is cluster x type.
    Rows sum to 1 when the mixture rows do (transfer_fix normalises them), so this is
    directly comparable to a fraction table.
    """
    M = mixture.reindex(leiden.values)                 # cells x types
    M.index = domains.values
    return M.groupby(level=0).mean()


def hard_composition(domains, labels):
    """Domain x label fraction table from hard per-cell labels (the baseline form)."""
    tab = pd.crosstab(domains, labels)
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def align(dfs):
    cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    return [d.reindex(columns=cols, fill_value=0.0) for d in dfs]


def matched_mismatched(c0, c1):
    """(matched_median, mismatched_mean, gap) between two domain x type tables.

    matched    = cos(c0[d], c1[d]) over shared domains
    mismatched = cos(c0[d], c1[d']) for d != d'
    The gap is invariant to a uniform density/sparsity shift affecting all pairs
    equally, which the raw matched cosine is not -- see the module docstring.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    dom = c0.index.intersection(c1.index)
    if len(dom) < 2:
        return np.nan, np.nan, np.nan
    A, B = align([c0.loc[dom], c1.loc[dom]])
    S = cosine_similarity(A.values, B.values)          # domains x domains
    matched = float(np.median(np.diag(S)))
    off = S[~np.eye(len(dom), dtype=bool)]
    mism = float(off.mean())
    return matched, mism, matched - mism


def main():
    p = argparse.ArgumentParser(description="NNLS-based niche comparison artifacts")
    p.add_argument("--results-dir", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--arms", default="scvi,scgpt")
    p.add_argument("--nnls-glob", default="xfer_nnls_{arm}_S{seed}",
                   help="run-dir template under --results-dir")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--baseline-annotation", default="annot_{arm}/FB080_{arm}_annotation.parquet",
                   help="published hard-label annotation, for the baseline comparison")
    p.add_argument("--save-compositions", action="store_true",
                   help="also write each weighted composition table")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    RES = Path(args.results_dir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = [r for r in args.regions.split(",") if r]
    methods = [m for m in args.methods.split(",") if m]
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    dom_col = f"domain_k{args.k}"

    # ---------- A. load mixtures + per-cell leiden for every (arm, seed) ----------
    log("A. loading NNLS mixtures and per-cell cluster assignments")
    mixtures, leidens, hard = {}, {}, {}
    for arm in arms:
        for s in seeds:
            d = RES / args.nnls_glob.format(arm=arm, seed=s)
            mp = d / f"FB080_{arm}_nnls_mixture.csv"
            ap = d / f"FB080_{arm}_nnls_annotation.parquet"
            if not mp.exists() or not ap.exists():
                log(f"   SKIP {arm} S{s}: missing {mp.name if not mp.exists() else ap.name}")
                continue
            mix = pd.read_csv(mp, index_col=0)
            mix.index = mix.index.astype(str)
            ann = pd.read_parquet(ap)
            mixtures[(arm, s)] = mix
            leidens[(arm, s)] = ann["leiden"].astype(str)
            hard[(arm, s)] = ann["annotation"].astype(str)
            log(f"   {arm} S{s}: mixture {mix.shape}, {len(ann)} cells, "
                f"{ann['annotation'].nunique()} distinct hard labels")
    if not mixtures:
        raise SystemExit("no NNLS runs loaded")

    # ---------- B. audit: what does the mixture recover that hard labels do not? ----------
    log("B. label-mass audit (hard argmax vs weighted mixture)")
    mass_rows = []
    for (arm, s), mix in sorted(mixtures.items()):
        lei, hl = leidens[(arm, s)], hard[(arm, s)]
        wmass = mix.reindex(lei.values).mean(axis=0)        # mean weight per type
        hmass = hl.value_counts(normalize=True)
        for t in mix.columns:
            mass_rows.append({
                "arm": arm, "seed": s, "label": t,
                "weighted_mass": round(float(wmass.get(t, 0.0)), 6),
                "hard_mass": round(float(hmass.get(t, 0.0)), 6),
                "recovered_only_by_weights": bool(wmass.get(t, 0.0) > 0.01
                                                  and hmass.get(t, 0.0) == 0.0),
            })
    mass = pd.DataFrame(mass_rows)
    mass.to_csv(outdir / "nnls_label_mass.csv", index=False)
    only = mass[mass["recovered_only_by_weights"]]
    log(f"   wrote nnls_label_mass.csv; {len(only)} (arm,seed,label) cases carry >1% "
        f"weighted mass but ZERO hard-label mass")
    for arm in arms:
        sub = sorted(only[only["arm"] == arm]["label"].unique())
        log(f"   [{arm}] recovered only by the weights: {', '.join(sub) if sub else '(none)'}")

    # ---------- C. annotation axis (Q1) under NNLS, over all seed pairs ----------
    log("C. annotation axis (Q1): matched vs mismatched composition cosine")
    rows = []
    for r in regions:
        for m in methods:
            dpath = RES / m / f"{r}_{m}_domains.parquet"
            if not dpath.exists():
                log(f"   SKIP {r}x{m}: no {dpath}")
                continue
            dom_all = pd.read_parquet(dpath)[dom_col]

            comps = {}
            for (arm, s), mix in mixtures.items():
                lei = leidens[(arm, s)]
                cells = dom_all.index.intersection(lei.index)
                comps[(arm, s)] = weighted_composition(
                    dom_all.loc[cells].astype(str), lei.loc[cells], mix)
                if args.save_compositions:
                    comps[(arm, s)].to_csv(
                        outdir / f"nnls_composition_{r}_{m}_{arm}_S{s}.csv")

            a0, a1 = arms[0], arms[1]
            for s0 in seeds:
                for s1 in seeds:
                    if (a0, s0) not in comps or (a1, s1) not in comps:
                        continue
                    mt, mm, gap = matched_mismatched(comps[(a0, s0)], comps[(a1, s1)])
                    rows.append({
                        "region": r, "method": m, "representation": "nnls_weighted",
                        f"{a0}_seed": s0, f"{a1}_seed": s1,
                        "matched_cosine": round(mt, 4), "mismatched_cosine": round(mm, 4),
                        "gap": round(gap, 4),
                    })

            # baseline hard-label arm, same statistic, for an apples-to-apples read
            try:
                b0 = pd.read_parquet(RES / args.baseline_annotation.format(arm=a0))
                b1 = pd.read_parquet(RES / args.baseline_annotation.format(arm=a1))
                c0cells = dom_all.index.intersection(b0.index)
                c1cells = dom_all.index.intersection(b1.index)
                bc0 = hard_composition(dom_all.loc[c0cells].astype(str),
                                       b0.loc[c0cells, "annotation"].astype(str))
                bc1 = hard_composition(dom_all.loc[c1cells].astype(str),
                                       b1.loc[c1cells, "annotation"].astype(str))
                mt, mm, gap = matched_mismatched(bc0, bc1)
                rows.append({
                    "region": r, "method": m, "representation": "baseline_hard",
                    f"{a0}_seed": -1, f"{a1}_seed": -1,
                    "matched_cosine": round(mt, 4), "mismatched_cosine": round(mm, 4),
                    "gap": round(gap, 4),
                })
            except FileNotFoundError as e:
                log(f"   baseline unavailable for {r}x{m}: {e}")
            log(f"   {r}x{m}: {len([x for x in rows if x['region'] == r and x['method'] == m])} rows")

    det = pd.DataFrame(rows)
    det.to_csv(outdir / "nnls_annotation_axis.csv", index=False)
    log(f"   wrote nnls_annotation_axis.csv ({len(det)} rows)")

    # ---------- D. summary with the seed error bar ----------
    log("D. summarising over seed pairs")
    srows = []
    for (r, m), g in det.groupby(["region", "method"]):
        nn = g[g["representation"] == "nnls_weighted"]
        bl = g[g["representation"] == "baseline_hard"]
        row = {"region": r, "method": m, "n_seed_pairs": len(nn)}
        for col in ("matched_cosine", "gap"):
            row[f"nnls_{col}_median"] = round(float(nn[col].median()), 4) if len(nn) else np.nan
            row[f"nnls_{col}_min"] = round(float(nn[col].min()), 4) if len(nn) else np.nan
            row[f"nnls_{col}_max"] = round(float(nn[col].max()), 4) if len(nn) else np.nan
            row[f"nnls_{col}_range"] = (round(float(nn[col].max() - nn[col].min()), 4)
                                        if len(nn) else np.nan)
            row[f"baseline_{col}"] = round(float(bl[col].iloc[0]), 4) if len(bl) else np.nan
        srows.append(row)
    summ = pd.DataFrame(srows)
    summ.to_csv(outdir / "nnls_annotation_axis_summary.csv", index=False)
    log(f"   wrote nnls_annotation_axis_summary.csv ({len(summ)} rows)")
    print(summ.to_string(index=False), flush=True)

    # ---------- E. cross-region V1-vs-V2 matching under NNLS ----------
    log("E. cross-region V1-vs-V2 mutual-best niche matching")
    from sklearn.metrics.pairwise import cosine_similarity
    xrows = []
    if len(regions) >= 2:
        r0, r1 = regions[0], regions[1]
        for m in methods:
            d0p, d1p = (RES / m / f"{r0}_{m}_domains.parquet",
                        RES / m / f"{r1}_{m}_domains.parquet")
            if not (d0p.exists() and d1p.exists()):
                continue
            dom0, dom1 = pd.read_parquet(d0p)[dom_col], pd.read_parquet(d1p)[dom_col]
            for arm in arms:
                for s in seeds:
                    if (arm, s) not in mixtures:
                        continue
                    mix, lei = mixtures[(arm, s)], leidens[(arm, s)]
                    c0 = weighted_composition(
                        dom0.loc[dom0.index.intersection(lei.index)].astype(str),
                        lei.loc[dom0.index.intersection(lei.index)], mix)
                    c1 = weighted_composition(
                        dom1.loc[dom1.index.intersection(lei.index)].astype(str),
                        lei.loc[dom1.index.intersection(lei.index)], mix)
                    A, B = align([c0, c1])
                    S = cosine_similarity(A.values, B.values)
                    best0, best1 = S.argmax(1), S.argmax(0)
                    mutual = [i for i in range(S.shape[0]) if best1[best0[i]] == i]
                    xrows.append({
                        "method": m, "arm": arm, "seed": s,
                        "n_domains": S.shape[0],
                        "n_mutual_best": len(mutual),
                        "n_mutual_best_ge_0.6": int(sum(S[i, best0[i]] >= 0.6 for i in mutual)),
                        "median_matched_cosine": round(float(np.median(
                            [S[i, best0[i]] for i in mutual])), 4) if mutual else np.nan,
                    })
    xr = pd.DataFrame(xrows)
    xr.to_csv(outdir / "nnls_crossregion_match.csv", index=False)
    log(f"   wrote nnls_crossregion_match.csv ({len(xr)} rows)")
    if len(xr):
        print(xr.groupby(["method", "arm"])[
            ["n_mutual_best", "n_mutual_best_ge_0.6", "median_matched_cosine"]
        ].agg(["mean", "min", "max"]).round(3).to_string(), flush=True)

    log("Reading it: cite the GAP, not matched_cosine -- NNLS composition vectors are "
        "dense over all 35 types while the baseline's were sparse, so the raw cosines "
        "are not comparable across representations; the matched-minus-mismatched gap "
        "cancels that. A gap near zero means the arms agree only because all domains "
        "look alike, however high the raw cosine.")
    log("DONE.")


if __name__ == "__main__":
    main()
