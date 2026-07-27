#!/usr/bin/env python
"""
Niche comparison harness for the V1-vs-V2 pipeline study. Consumes profile_niches
outputs (composition / signaling / profile) + per-cell domain parquets across the
2x2 = {annotations} x {spatial methods} x {regions}, and produces:

  A. CROSS-METHOD (spatial axis; same region, same cells): Jaccard of domain cell-
     membership between the two spatial methods -> ROBUST vs METHOD-SPECIFIC niches
     + partition ARI. Domains are annotation-independent, so this is computed once.

  B. CROSS-REGION (V1 vs V2; separate pieces): match niches by COMPOSITION + LAYER
     cosine (signaling held out); mutual-best above threshold = matched, else
     region-specific. Run PER annotation arm to see if the V1-vs-V2 story is stable.
     For matched pairs, report top differential signaling pathways.

  C. ANNOTATION AXIS (Q1; scVI vs scGPT on the SAME domains): per (region, method),
     compare each domain's cell-type composition between annotation arms (cosine +
     L1 + top cell-type shifts) -> does the FM embedding change niche composition?

Runs in the neuro env; tables are small.
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ari_jaccard(d0, d1, jaccard_thr):
    """d0, d1: aligned per-cell domain-label Series. Returns (ari, cont, jac, best_df)."""
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(d0.values, d1.values)
    cont = pd.crosstab(d0, d1)
    jac = cont / (cont.sum(1).values[:, None] + cont.sum(0).values[None, :] - cont)
    best = pd.DataFrame({
        "m0_domain": jac.index, "m1_best": jac.idxmax(1).values,
        "jaccard": jac.max(1).values.round(3),
    })
    best["robust"] = best["jaccard"] >= jaccard_thr
    return ari, cont, jac, best


def _permutation_pvalue_ari(d0, d1, observed_ari, n_perm, rng):
    """Null: permute d1's per-cell labels (breaks cell correspondence, keeps label
    frequencies), recompute ARI n_perm times. One-sided empirical p-value (is the
    observed agreement higher than chance)."""
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    d1_vals = d1.values
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = adjusted_rand_score(d0.values, rng.permutation(d1_vals))
    pval = (1 + int(np.sum(null >= observed_ari))) / (n_perm + 1)
    return pval, null


def main():
    p = argparse.ArgumentParser(description="Cross-method + cross-region + annotation-axis niche matching")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--annotations", default="scvi,scgpt")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--kmin", type=int, default=6, help="k-sweep lower bound (robustness-across-k check)")
    p.add_argument("--kmax", type=int, default=14, help="k-sweep upper bound (robustness-across-k check)")
    p.add_argument("--n-perm", type=int, default=1000, help="permutations for domain-agreement null (Section A)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jaccard-thr", type=float, default=0.25)
    p.add_argument("--match-thr", type=float, default=0.6)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.default_rng(args.seed)

    RES = Path(args.results)
    anns = args.annotations.split(",")
    methods = args.methods.split(",")
    regions = args.regions.split(",")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def ndir(r, m, a, k=None): return RES / f"niche_{r}_{m}K{args.k if k is None else k}_{a}"
    def comp(r, m, a, k=None): return pd.read_csv(ndir(r, m, a, k) / f"{r}_niche_composition.csv", index_col=0)
    def prof(r, m, a, k=None): return pd.read_csv(ndir(r, m, a, k) / f"{r}_niche_profile.csv", index_col=0)
    def sig(r, m, a, k=None): return pd.read_csv(ndir(r, m, a, k) / f"{r}_niche_signaling.csv", index_col=0)
    def dom(r, m, k=None): return pd.read_parquet(RES / m / f"{r}_{m}_domains.parquet")[f"domain_k{args.k if k is None else k}"]

    def align(dfs):
        cols = sorted(set().union(*[d.columns for d in dfs]))
        return [d.reindex(columns=cols, fill_value=0.0) for d in dfs]

    # ---------- A. cross-method (annotation-independent) ----------
    log("A. CROSS-METHOD (Jaccard of domain cell-membership; annotation-independent)")
    if len(methods) >= 2:
        m0, m1 = methods[:2]
        for r in regions:
            d0 = dom(r, m0); d1 = dom(r, m1).reindex(d0.index)
            cont = pd.crosstab(d0, d1)
            jac = cont / (cont.sum(1).values[:, None] + cont.sum(0).values[None, :] - cont)
            jac.to_csv(outdir / f"{r}_crossmethod_jaccard.csv")
            best = pd.DataFrame({f"{m0}": jac.index, f"{m1}_best": jac.idxmax(1).values,
                                 "jaccard": jac.max(1).values.round(3)})
            best["robust"] = best["jaccard"] >= args.jaccard_thr
            best.to_csv(outdir / f"{r}_crossmethod_matches.csv", index=False)
            log(f"  {r}: ARI={adjusted_rand_score(d0.values, d1.values):.3f} | "
                f"robust {int(best['robust'].sum())}/{len(best)}, method-specific {int((~best['robust']).sum())}")

        # ---- A2. k-robustness + permutation significance (sweeps args.kmin..kmax) ----
        log(f"A2. k-robustness sweep (k={args.kmin}..{args.kmax}) + permutation p-values "
            f"(n_perm={args.n_perm}, seed={args.seed})")
        by_k_rows = []
        for r in regions:
            for k in range(args.kmin, args.kmax + 1):
                d0 = dom(r, m0, k); d1 = dom(r, m1, k).reindex(d0.index)
                ari, cont, jac, best = _ari_jaccard(d0, d1, args.jaccard_thr)
                pval, _null = _permutation_pvalue_ari(d0, d1, ari, args.n_perm, rng)
                by_k_rows.append({
                    "region": r, "k": k, "ari": round(ari, 4), "p_value_ari": round(pval, 4),
                    "jaccard_matched_frac_at_thr": round(float(best["robust"].mean()), 4),
                    "n_domains_method0": int(d0.nunique()), "n_domains_method1": int(d1.nunique()),
                })
        by_k = pd.DataFrame(by_k_rows)
        by_k.to_csv(outdir / "crossmethod_ari_by_k.csv", index=False)
        log(f"  wrote crossmethod_ari_by_k.csv ({len(by_k)} rows); "
            f"ARI range [{by_k['ari'].min():.3f}, {by_k['ari'].max():.3f}], "
            f"all p<0.05: {bool((by_k['p_value_ari'] < 0.05).all())}")

    # ---------- B. cross-region, per annotation arm ----------
    log("B. CROSS-REGION V1-vs-V2 (composition+layer cosine; per annotation arm)")
    r0, r1 = regions[:2]
    for a in anns:
        for m in methods:
            c0, c1 = align([comp(r0, m, a), comp(r1, m, a)])
            l0, l1 = align([prof(r0, m, a).filter(like="layer_"), prof(r1, m, a).filter(like="layer_")])
            F0 = np.hstack([c0.values, l0.reindex(c0.index).values])
            F1 = np.hstack([c1.values, l1.reindex(c1.index).values])
            S = cosine_similarity(F0, F1)
            r0b, r1b = S.argmax(1), S.argmax(0)
            rows = [{f"{r0}_niche": c0.index[i], f"{r1}_niche": c1.index[int(r0b[i])],
                     "cosine": round(float(S[i, int(r0b[i])]), 3),
                     "matched": bool(int(r1b[int(r0b[i])]) == i and S[i, int(r0b[i])] >= args.match_thr)}
                    for i in range(len(c0))]
            mt = pd.DataFrame(rows); mt.to_csv(outdir / f"{m}_{a}_{r0}{r1}_matches.csv", index=False)
            nm = int(mt["matched"].sum())
            log(f"  {a} x {m}: matched {nm} | {r0}-specific {len(mt)-nm} | "
                f"{r1}-specific {len(c1)-len(set(mt[mt['matched']][f'{r1}_niche']))}")

    # ---------- B2. cross-region k-robustness sweep (needs per-k profile_niches outputs) ----------
    log(f"B2. cross-region k-robustness sweep (k={args.kmin}..{args.kmax}); "
        f"requires niche_{{r}}_{{m}}K{{k}}_{{a}} dirs to exist per k")
    b_rows = []
    for a in anns:
        for m in methods:
            for k in range(args.kmin, args.kmax + 1):
                if not (ndir(r0, m, a, k) / f"{r0}_niche_composition.csv").exists() or \
                   not (ndir(r1, m, a, k) / f"{r1}_niche_composition.csv").exists():
                    log(f"  skip {a} x {m} k={k}: niche_{{}}_{{}}K{{}}_{{}} dir(s) not found yet".format(r0, m, k, a))
                    continue
                c0, c1 = align([comp(r0, m, a, k), comp(r1, m, a, k)])
                l0, l1 = align([prof(r0, m, a, k).filter(like="layer_"), prof(r1, m, a, k).filter(like="layer_")])
                F0 = np.hstack([c0.values, l0.reindex(c0.index).values])
                F1 = np.hstack([c1.values, l1.reindex(c1.index).values])
                S = cosine_similarity(F0, F1)
                r0b, r1b = S.argmax(1), S.argmax(0)
                matched = sum(bool(int(r1b[int(r0b[i])]) == i and S[i, int(r0b[i])] >= args.match_thr)
                              for i in range(len(c0)))
                r1_matched_niches = {c1.index[int(r0b[i])] for i in range(len(c0))
                                     if int(r1b[int(r0b[i])]) == i and S[i, int(r0b[i])] >= args.match_thr}
                b_rows.append({
                    "annotation": a, "method": m, "k": k, "region0": r0, "region1": r1,
                    "n_niches": len(c0), "n_matched": matched,
                    "n_region0_specific": len(c0) - matched,
                    "n_region1_specific": len(c1) - len(r1_matched_niches),
                })
    if b_rows:
        b_by_k = pd.DataFrame(b_rows)
        b_by_k.to_csv(outdir / "crossregion_match_by_k.csv", index=False)
        log(f"  wrote crossregion_match_by_k.csv ({len(b_by_k)} rows across {b_by_k['k'].nunique()} k values)")
    else:
        log("  no per-k niche dirs found beyond k (skipped entirely) -- crossregion_match_by_k.csv not written")

    # ---------- C. annotation axis (Q1): scVI vs scGPT composition, same domains ----------
    if len(anns) >= 2:
        a0, a1 = anns[:2]
        log(f"C. ANNOTATION AXIS ({a0} vs {a1}; same domains -> composition change = Q1)")
        for r in regions:
            for m in methods:
                c0, c1 = align([comp(r, m, a0), comp(r, m, a1)])
                dom_ids = list(c0.index)
                cos = [float(cosine_similarity(c0.loc[[d]].values, c1.loc[[d]].values)[0, 0]) for d in dom_ids]
                l1d = [float(np.abs(c0.loc[d].values - c1.loc[d].values).sum()) for d in dom_ids]
                shifts = []
                for d in dom_ids:
                    diff = (c1.loc[d] - c0.loc[d])
                    top = diff.reindex(diff.abs().sort_values(ascending=False).index).head(3)
                    shifts.append("; ".join(f"{k}{v:+.2f}" for k, v in top.items()))
                out = pd.DataFrame({"domain": dom_ids, "composition_cosine": np.round(cos, 3),
                                    "L1_distance": np.round(l1d, 3), "top_celltype_shifts": shifts})
                out.to_csv(outdir / f"{r}_{m}_annaxis_{a0}_vs_{a1}.csv", index=False)
                log(f"  {r} x {m}: composition cosine ({a0}~{a1}) median {np.median(cos):.3f} "
                    f"[min {np.min(cos):.3f}]; median L1 {np.median(l1d):.3f}")

        # ---- C2. annotation-axis k-robustness sweep (needs per-k profile_niches outputs) ----
        log(f"C2. annotation-axis k-robustness sweep (k={args.kmin}..{args.kmax})")
        c_rows = []
        for r in regions:
            for m in methods:
                for k in range(args.kmin, args.kmax + 1):
                    if not (ndir(r, m, a0, k) / f"{r}_niche_composition.csv").exists() or \
                       not (ndir(r, m, a1, k) / f"{r}_niche_composition.csv").exists():
                        log(f"  skip {r} x {m} k={k}: niche_{r}_{m}K{k}_{{{a0}|{a1}}} dir(s) not found yet")
                        continue
                    c0, c1 = align([comp(r, m, a0, k), comp(r, m, a1, k)])
                    dom_ids = list(c0.index)
                    cos = [float(cosine_similarity(c0.loc[[d]].values, c1.loc[[d]].values)[0, 0]) for d in dom_ids]
                    l1d = [float(np.abs(c0.loc[d].values - c1.loc[d].values).sum()) for d in dom_ids]
                    c_rows.append({
                        "region": r, "method": m, "k": k, "annotation0": a0, "annotation1": a1,
                        "n_domains": len(dom_ids),
                        "cosine_median": round(float(np.median(cos)), 4),
                        "cosine_min": round(float(np.min(cos)), 4),
                        "L1_median": round(float(np.median(l1d)), 4),
                    })
        if c_rows:
            c_by_k = pd.DataFrame(c_rows)
            c_by_k.to_csv(outdir / "annotation_axis_cosine_by_k.csv", index=False)
            log(f"  wrote annotation_axis_cosine_by_k.csv ({len(c_by_k)} rows across {c_by_k['k'].nunique()} k values)")
        else:
            log("  no per-k niche dirs found beyond k -- annotation_axis_cosine_by_k.csv not written")

    log(f"DONE. wrote cross-method + cross-region(per-arm) + annotation-axis comparisons to {outdir}")


if __name__ == "__main__":
    main()
