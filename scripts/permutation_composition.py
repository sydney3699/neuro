#!/usr/bin/env python
"""
Permutation-based significance testing for the composition-based niche
comparisons in compare_niches.py (Section B: cross-region matching; Section C:
scVI-vs-scGPT annotation axis). Those comparisons operate on per-domain
composition CSVs already aggregated by profile_niches.py; a real null needs
per-cell data, so this script re-derives composition directly from the same
per-cell domain-parquet + annotation-parquet inputs profile_niches.py reads
(domain-parquet: {region}_{method}_domains.parquet['domain_k{k}']; annotation-
parquet: FB080_{annotation}_annotation.parquet['annotation'], joined by cell id).

SECTION C NULL (annotation axis): for each (region, method, k), the domain
assignment is identical for both annotation arms (verified by assertion).
Independently permute each arm's per-cell labels (within region) to destroy any
true domain<->cell-type relationship while preserving each arm's overall label-
frequency marginal and domain sizes. Recompute per-domain composition for both
arms under permutation, recompute cosine similarity per domain, repeat
--n-perm times. This null answers: "if domain assignment carried no true cell-
type signal beyond marginal label frequency, how similar would the two arms'
domain compositions look purely by chance?" p_value = fraction(null_cosine <=
observed_cosine): a small p means the observed cross-arm disagreement in that
domain is more extreme than chance -- real signal that the FM choice changes
composition, not noise from finite-sample marginals.

SECTION B NULL (cross-region matching): for each (annotation, method), permute
each region's per-cell domain labels independently (destroying the true
domain-composition relationship while preserving domain-size marginals),
recompute composition feature matrices, recompute the max-cosine cross-region
match value for each niche, repeat --n-perm times. p_value = fraction(
null_cosine >= observed_cosine): a small p means the observed cross-region
match is closer than chance, i.e. a real correspondence rather than an
artifact of domain-size overlap.

NOTE: Section B's null uses composition only (not the layer_* features
compare_niches.py also blends in) -- layer fractions aren't well-defined under
a permuted domain assignment without re-deriving them from obs, and cell-type
composition is the more direct thing to permutation-test here.
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cells(results, region, method, k, annotation):
    import pandas as pd
    dom_path = Path(results) / method / f"{region}_{method}_domains.parquet"
    dom_df = pd.read_parquet(dom_path)
    domain = dom_df[f"domain_k{k}"].astype(str)
    ann_path = Path(results) / f"annot_{annotation}" / f"FB080_{annotation}_annotation.parquet"
    ann_df = pd.read_parquet(ann_path)
    celltype = ann_df["annotation"].reindex(domain.index).astype(str)
    if celltype.isna().any():
        raise SystemExit(f"{celltype.isna().sum()} of {len(celltype)} cells missing annotation "
                          f"for {region}/{annotation} ({ann_path})")
    return domain, celltype


def composition(domain, celltype):
    import pandas as pd
    comp = pd.crosstab(domain, celltype)
    return comp.div(comp.sum(axis=1), axis=0)


def align(dfs):
    cols = sorted(set().union(*[d.columns for d in dfs]))
    return [d.reindex(columns=cols, fill_value=0.0) for d in dfs]


def annotation_axis_pvalues(results, region, method, k, a0, a1, n_perm, rng, outdir):
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    dom0, ct0 = load_cells(results, region, method, k, a0)
    dom1, ct1 = load_cells(results, region, method, k, a1)
    if not (dom0.index.equals(dom1.index) and (dom0.values == dom1.values).all()):
        raise SystemExit(f"domain assignment differs between annotation arms for {region}/{method}/k{k} "
                          f"(expected identical -- domain depends only on region+method+k)")
    domain = dom0

    c0_obs, c1_obs = align([composition(domain, ct0), composition(domain, ct1)])
    dom_ids = list(c0_obs.index)
    observed = {d: float(cosine_similarity(c0_obs.loc[[d]].values, c1_obs.loc[[d]].values)[0, 0]) for d in dom_ids}

    null = {d: np.full(n_perm, np.nan) for d in dom_ids}
    ct0_vals = ct0.values.copy()
    ct1_vals = ct1.values.copy()
    for i in range(n_perm):
        p0 = pd.Series(rng.permutation(ct0_vals), index=domain.index)
        p1 = pd.Series(rng.permutation(ct1_vals), index=domain.index)
        c0p, c1p = align([composition(domain, p0), composition(domain, p1)])
        for d in dom_ids:
            if d in c0p.index and d in c1p.index:
                null[d][i] = float(cosine_similarity(c0p.loc[[d]].values, c1p.loc[[d]].values)[0, 0])

    rows = []
    for d in dom_ids:
        nd = null[d][~np.isnan(null[d])]
        pval = (1 + np.sum(nd <= observed[d])) / (len(nd) + 1) if len(nd) else float("nan")
        rows.append({"domain": d, "observed_cosine": round(observed[d], 4),
                     "null_median": round(float(np.median(nd)), 4) if len(nd) else float("nan"),
                     "null_p05": round(float(np.percentile(nd, 5)), 4) if len(nd) else float("nan"),
                     "p_value": round(float(pval), 4) if pval == pval else float("nan"),
                     "n_perm_valid": int(len(nd))})
    out = pd.DataFrame(rows)
    fn = Path(outdir) / f"{region}_{method}_annaxis_{a0}_vs_{a1}_k{k}_pvalues.csv"
    out.to_csv(fn, index=False)
    log(f"  {region} x {method}: annotation-axis p-values -> {fn.name} "
        f"(median p={out['p_value'].median():.3f})")
    return out


def crossregion_pvalues(results, r0, r1, method, annotation, k, n_perm, rng, match_thr, outdir):
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    dom0, ct0 = load_cells(results, r0, method, k, annotation)
    dom1, ct1 = load_cells(results, r1, method, k, annotation)

    c0_obs, c1_obs = align([composition(dom0, ct0), composition(dom1, ct1)])
    S_obs = cosine_similarity(c0_obs.values, c1_obs.values)
    best_idx = S_obs.argmax(1)
    observed = {c0_obs.index[i]: float(S_obs[i, best_idx[i]]) for i in range(len(c0_obs))}
    best_match = {c0_obs.index[i]: c1_obs.index[best_idx[i]] for i in range(len(c0_obs))}

    null = {d: np.full(n_perm, np.nan) for d in c0_obs.index}
    dom0_vals = dom0.values.copy()
    dom1_vals = dom1.values.copy()
    for i in range(n_perm):
        pdom0 = pd.Series(rng.permutation(dom0_vals), index=dom0.index)
        pdom1 = pd.Series(rng.permutation(dom1_vals), index=dom1.index)
        c0p, c1p = align([composition(pdom0, ct0), composition(pdom1, ct1)])
        Sp = cosine_similarity(c0p.values, c1p.values)
        for d in c0_obs.index:
            if d in c0p.index:
                row = c0p.index.get_loc(d)
                null[d][i] = float(Sp[row].max())

    rows = []
    for d in c0_obs.index:
        nd = null[d][~np.isnan(null[d])]
        pval = (1 + np.sum(nd >= observed[d])) / (len(nd) + 1) if len(nd) else float("nan")
        rows.append({f"{r0}_niche": d, "best_match": best_match[d],
                     "observed_cosine": round(observed[d], 4),
                     "null_median": round(float(np.median(nd)), 4) if len(nd) else float("nan"),
                     "null_p95": round(float(np.percentile(nd, 95)), 4) if len(nd) else float("nan"),
                     "p_value": round(float(pval), 4) if pval == pval else float("nan"),
                     "matched_thr": observed[d] >= match_thr})
    out = pd.DataFrame(rows)
    fn = Path(outdir) / f"{method}_{annotation}_{r0}{r1}_crossregion_k{k}_pvalues.csv"
    out.to_csv(fn, index=False)
    log(f"  {method} x {annotation}: cross-region p-values -> {fn.name} "
        f"(median p={out['p_value'].median():.3f})")
    return out


def main():
    p = argparse.ArgumentParser(description="Permutation significance for composition-based niche comparisons")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--annotations", default="scvi,scgpt")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--match-thr", type=float, default=0.6)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--skip-annaxis", action="store_true")
    p.add_argument("--skip-crossregion", action="store_true")
    args = p.parse_args()

    import numpy as np

    regions = args.regions.split(",")
    methods = args.methods.split(",")
    anns = args.annotations.split(",")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if not args.skip_annaxis and len(anns) >= 2:
        a0, a1 = anns[:2]
        log(f"ANNOTATION AXIS null ({a0} vs {a1}), n_perm={args.n_perm}, k={args.k}")
        for r in regions:
            for m in methods:
                annotation_axis_pvalues(args.results, r, m, args.k, a0, a1, args.n_perm, rng, outdir)

    if not args.skip_crossregion and len(regions) >= 2:
        r0, r1 = regions[:2]
        log(f"CROSS-REGION null ({r0} vs {r1}), n_perm={args.n_perm}, k={args.k}")
        for a in anns:
            for m in methods:
                crossregion_pvalues(args.results, r0, r1, m, a, args.k, args.n_perm, rng, args.match_thr, outdir)

    log(f"DONE. wrote permutation p-values to {outdir}")


if __name__ == "__main__":
    main()
