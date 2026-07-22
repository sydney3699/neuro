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


def main():
    p = argparse.ArgumentParser(description="Cross-method + cross-region + annotation-axis niche matching")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--annotations", default="scvi,scgpt")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--jaccard-thr", type=float, default=0.25)
    p.add_argument("--match-thr", type=float, default=0.6)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.pairwise import cosine_similarity

    RES = Path(args.results)
    anns = args.annotations.split(",")
    methods = args.methods.split(",")
    regions = args.regions.split(",")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def ndir(r, m, a): return RES / f"niche_{r}_{m}K{args.k}_{a}"
    def comp(r, m, a): return pd.read_csv(ndir(r, m, a) / f"{r}_niche_composition.csv", index_col=0)
    def prof(r, m, a): return pd.read_csv(ndir(r, m, a) / f"{r}_niche_profile.csv", index_col=0)
    def sig(r, m, a): return pd.read_csv(ndir(r, m, a) / f"{r}_niche_signaling.csv", index_col=0)
    def dom(r, m): return pd.read_parquet(RES / m / f"{r}_{m}_domains.parquet")[f"domain_k{args.k}"]

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

    log(f"DONE. wrote cross-method + cross-region(per-arm) + annotation-axis comparisons to {outdir}")


if __name__ == "__main__":
    main()
