#!/usr/bin/env python
"""
Niche comparison harness for the V1-vs-V2 pipeline study. Consumes profile_niches
outputs (composition / signaling / profile) + the per-cell domain parquets, and
produces the two matching analyses the design calls for:

  A. CROSS-METHOD (same region, same cells, different partitions): Jaccard on
     cell membership between the two spatial methods' domains -> which niches are
     ROBUST (found by both methods) vs METHOD-SPECIFIC. Primary metric = Jaccard
     (spatial overlap), NOT signaling. Also reports the partition-level ARI.

  B. CROSS-REGION (V1 vs V2, separate tissue pieces -> no shared space): match
     niches by COMPOSITION + LAYER similarity (cosine of [cell-type fractions ++
     layer fractions]); mutual-best above threshold = matched, else region-
     specific (its own category). Signaling is held out as an independent readout:
     for matched V1<->V2 niches, report the top differential pathways.

Runs in the neuro env (pandas/sklearn); tables are small. Add pipeline cells
(e.g. scGPT arm) by extending --methods/--annotations; the same matching applies.
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Cross-method + cross-region niche matching")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--annotation", default="scvi", help="annotation tag in the niche dir name")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--jaccard-thr", type=float, default=0.25, help="cross-method: >=this = robust niche")
    p.add_argument("--match-thr", type=float, default=0.6, help="cross-region: composition+layer cosine >=this to match")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.pairwise import cosine_similarity

    RES = Path(args.results)
    methods = args.methods.split(",")
    regions = args.regions.split(",")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def ndir(r, m): return RES / f"niche_{r}_{m}K{args.k}_{args.annotation}"
    def comp(r, m): return pd.read_csv(ndir(r, m) / f"{r}_niche_composition.csv", index_col=0)
    def prof(r, m): return pd.read_csv(ndir(r, m) / f"{r}_niche_profile.csv", index_col=0)
    def sig(r, m): return pd.read_csv(ndir(r, m) / f"{r}_niche_signaling.csv", index_col=0)
    def dom(r, m): return pd.read_parquet(RES / m / f"{r}_{m}_domains.parquet")[f"domain_k{args.k}"]

    # ---------- A. cross-method matching (per region) ----------
    log("A. CROSS-METHOD niche matching (Jaccard of domain cell-membership)")
    for r in regions:
        if len(methods) < 2:
            break
        m0, m1 = methods[0], methods[1]
        d0, d1 = dom(r, m0), dom(r, m1).reindex(dom(r, m0).index)
        cont = pd.crosstab(d0, d1)
        n0, n1 = cont.sum(1), cont.sum(0)
        jac = cont / (n0.values[:, None] + n1.values[None, :] - cont)
        jac.to_csv(outdir / f"{r}_crossmethod_jaccard.csv")
        best = pd.DataFrame({f"{m0}_niche": jac.index,
                             f"{m1}_best": jac.idxmax(1).values,
                             "jaccard": jac.max(1).values.round(3)})
        best["robust"] = best["jaccard"] >= args.jaccard_thr
        best.to_csv(outdir / f"{r}_crossmethod_matches.csv", index=False)
        ari = adjusted_rand_score(d0.values, d1.values)
        nrob = int(best["robust"].sum())
        log(f"  {r}: ARI({m0},{m1})={ari:.3f} | robust niches {nrob}/{len(best)} "
            f"(Jaccard>={args.jaccard_thr}); method-specific {len(best)-nrob}")

    # ---------- B. cross-region matching (per method) ----------
    log("B. CROSS-REGION niche matching (composition + layer cosine; signaling held out)")
    for m in methods:
        r0, r1 = regions[0], regions[1]
        c0, c1 = comp(r0, m), comp(r1, m)
        ccols = sorted(set(c0.columns) | set(c1.columns))
        c0 = c0.reindex(columns=ccols, fill_value=0.0); c1 = c1.reindex(columns=ccols, fill_value=0.0)
        l0 = prof(r0, m).filter(like="layer_"); l1 = prof(r1, m).filter(like="layer_")
        lcols = sorted(set(l0.columns) | set(l1.columns))
        l0 = l0.reindex(columns=lcols, fill_value=0.0); l1 = l1.reindex(columns=lcols, fill_value=0.0)
        F0 = np.hstack([c0.values, l0.reindex(c0.index).values])
        F1 = np.hstack([c1.values, l1.reindex(c1.index).values])
        S = cosine_similarity(F0, F1)                        # r0 niches x r1 niches
        r0_best, r1_best = S.argmax(1), S.argmax(0)
        rows = []
        for i, n0 in enumerate(c0.index):
            j = int(r0_best[i])
            mutual = int(r1_best[j]) == i
            rows.append({f"{r0}_niche": n0, f"{r1}_niche": c1.index[j],
                         "cosine": round(float(S[i, j]), 3),
                         "matched": bool(mutual and S[i, j] >= args.match_thr)})
        matches = pd.DataFrame(rows)
        matches.to_csv(outdir / f"{m}_{r0}{r1}_matches.csv", index=False)
        matched = matches[matches["matched"]]
        r1_matched = set(matched[f"{r1}_niche"])
        n_r0_spec = (~matches["matched"]).sum()
        n_r1_spec = sum(n not in r1_matched for n in c1.index)
        log(f"  {m}: matched {len(matched)} | {r0}-specific {n_r0_spec} | {r1}-specific {n_r1_spec}")

        # signaling differential for matched niches (independent readout)
        s0, s1 = sig(r0, m), sig(r1, m)
        scols = [c for c in s0.columns if c in s1.columns]
        drows = []
        for _, row in matched.iterrows():
            a = s0.loc[row[f"{r0}_niche"], scols].astype(float)
            b = s1.loc[row[f"{r1}_niche"], scols].astype(float)
            diff = (b - a).sort_values()
            up = "; ".join(f"{k}{v:+.2f}" for k, v in diff.tail(4)[::-1].items())     # r1 > r0
            dn = "; ".join(f"{k}{v:+.2f}" for k, v in diff.head(4).items())            # r0 > r1
            drows.append({f"{r0}_niche": row[f"{r0}_niche"], f"{r1}_niche": row[f"{r1}_niche"],
                          f"{r1}_up": up, f"{r0}_up": dn})
        if drows:
            pd.DataFrame(drows).to_csv(outdir / f"{m}_{r0}{r1}_signaling_diff.csv", index=False)

    log(f"DONE. wrote cross-method (jaccard/matches) + cross-region (matches/signaling_diff) to {outdir}")


if __name__ == "__main__":
    main()
