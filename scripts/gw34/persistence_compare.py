#!/usr/bin/env python
"""GW34 Stage 5 — GW20-vs-GW34 persistence comparison.

Three views of "do the V1-vs-V2 differences persist from GW20 to GW34?":
 1. SIGNALING (primary, harmonization-free): correlate the per-pathway V1-vs-V2
    effect sizes (rank_biserial from signaling_diff.py) between timepoints; classify
    each pathway/direction as persistent / gained / lost / reversed. WNT highlighted.
 2. COMPOSITION: per common H1 class, the V1-minus-V2 fraction at each timepoint;
    correlate the V1-vs-V2 composition signature across timepoints.
 3. NICHE: match GW34 STAGATE niches to GW20 niches (same region, cross-timepoint)
    by per-domain composition(H1)+signaling cosine — which niches recur.

Each section is independent and skipped if its inputs are absent. Runs in the neuro env.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from neurospatial.metrics import classify


def log(m):
    print(m, flush=True)


def signaling_persistence(gw20_csv, gw34_csv, outdir, eff_thr, q_thr):
    if not (Path(gw20_csv).exists() and Path(gw34_csv).exists()):
        log(f"[signaling] SKIP: need {gw20_csv} and {gw34_csv}"); return
    a = pd.read_csv(gw20_csv).set_index("column")
    b = pd.read_csv(gw34_csv).set_index("column")
    common = a.index.intersection(b.index)
    m = pd.DataFrame({
        "pathway": a.loc[common, "pathway"], "direction": a.loc[common, "direction"],
        "rb_gw20": a.loc[common, "rank_biserial"], "rb_gw34": b.loc[common, "rank_biserial"],
        "diff_gw20": a.loc[common, "diff"], "diff_gw34": b.loc[common, "diff"],
        "q_gw20": a.loc[common, "q_value"], "q_gw34": b.loc[common, "q_value"],
    })
    m["persistence"] = [classify(r.rb_gw20, r.rb_gw34, r.q_gw20, r.q_gw34, eff_thr, q_thr)
                        for r in m.itertuples()]
    m = m.sort_values("persistence")
    m.to_csv(outdir / "persistence_signaling.csv")
    rho = m[["rb_gw20", "rb_gw34"]].corr(method="spearman").iloc[0, 1]
    log(f"[signaling] {len(m)} pathway-directions | Spearman(rb_gw20, rb_gw34) = {rho:.3f}")
    log("  persistence classes:\n" + m["persistence"].value_counts().to_string())
    wnt = m[m["pathway"] == "WNT"]
    if len(wnt):
        log("  WNT:\n" + wnt[["direction", "rb_gw20", "rb_gw34", "q_gw20", "q_gw34", "persistence"]].to_string(index=False))


def composition_persistence(harm_dir, regions, outdir):
    harm_dir = Path(harm_dir)
    rows = {}
    for stage in ("gw20", "gw34"):
        for r in regions:
            f = harm_dir / f"{stage}_{r}_h1harm.parquet"
            if not f.exists():
                log(f"[composition] SKIP {stage} {r}: {f.name} missing"); return
            frac = pd.read_parquet(f)["annotation"].value_counts(normalize=True)
            rows[(stage, r)] = frac
    classes = sorted(set().union(*[s.index for s in rows.values()]))
    tab = pd.DataFrame({f"{st}_{r}": rows[(st, r)].reindex(classes).fillna(0.0)
                        for (st, r) in rows}, index=classes)
    r0, r1 = regions[:2]
    tab["gw20_v1_minus_v2"] = tab[f"gw20_{r0}"] - tab[f"gw20_{r1}"]
    tab["gw34_v1_minus_v2"] = tab[f"gw34_{r0}"] - tab[f"gw34_{r1}"]
    tab.to_csv(outdir / "persistence_composition.csv")
    # correlation of the V1-vs-V2 composition signature across timepoints (shared mature classes)
    shared = [c for c in classes if c != "Other-progenitor"]
    rho = tab.loc[shared, ["gw20_v1_minus_v2", "gw34_v1_minus_v2"]].corr(method="spearman").iloc[0, 1]
    log(f"[composition] classes={classes}")
    log(f"  V1-minus-V2 signature Spearman (GW20 vs GW34, shared mature classes) = {rho:.3f}")
    log("  table:\n" + tab.round(3).to_string())


def niche_persistence(gw20_niche_dir, gw34_niche_dir, regions, outdir, match_thr):
    gw20_niche_dir, gw34_niche_dir = Path(gw20_niche_dir), Path(gw34_niche_dir)
    out_rows = []
    for r in regions:
        f20 = gw20_niche_dir / f"{r}_niche_composition.csv"
        f34 = gw34_niche_dir / f"{r}_niche_composition.csv"
        if not (f20.exists() and f34.exists()):
            log(f"[niche] SKIP {r}: need {f20} and {f34}"); continue
        c20 = pd.read_csv(f20, index_col=0)
        c34 = pd.read_csv(f34, index_col=0)
        cols = sorted(set(c20.columns) | set(c34.columns))
        C20 = c20.reindex(columns=cols, fill_value=0.0)
        C34 = c34.reindex(columns=cols, fill_value=0.0)
        S = cosine_similarity(C34.values, C20.values)  # rows=gw34 domains, cols=gw20
        best = S.argmax(1)
        back = S.argmax(0)
        for i, dom34 in enumerate(C34.index):
            j = int(best[i])
            mutual = int(back[j]) == i
            out_rows.append({
                "region": r, "gw34_domain": dom34, "gw20_best_match": C20.index[j],
                "cosine": round(float(S[i, j]), 3),
                "matched": bool(mutual and S[i, j] >= match_thr),
            })
    if out_rows:
        df = pd.DataFrame(out_rows)
        df.to_csv(outdir / "persistence_niches.csv", index=False)
        n_match = int(df["matched"].sum())
        log(f"[niche] {len(df)} GW34 domains scored | {n_match} mutual-best matched to a GW20 niche (thr={match_thr})")
        log(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="GW20-vs-GW34 persistence comparison")
    p.add_argument("--gw20-signaling", default="/scratch/cole.sy/neuro/results/signaling_diff/signaling_diff_cpmz.csv")
    p.add_argument("--gw34-signaling", default="/scratch/cole.sy/neuro/results/signaling_diff_gw34/signaling_diff_cpmz.csv")
    p.add_argument("--harm-dir", default="/scratch/cole.sy/neuro/results/h1harm")
    p.add_argument("--gw20-niche-dir", default="/scratch/cole.sy/neuro/results/niche_gw20_stagate_K8_h1")
    p.add_argument("--gw34-niche-dir", default="/scratch/cole.sy/neuro/results/niche_gw34_stagate_K8")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--eff-thr", type=float, default=0.05, help="min |rank_biserial| for 'significant effect'")
    p.add_argument("--q-thr", type=float, default=0.05)
    p.add_argument("--match-thr", type=float, default=0.6)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = args.regions.split(",")
    signaling_persistence(args.gw20_signaling, args.gw34_signaling, outdir, args.eff_thr, args.q_thr)
    composition_persistence(args.harm_dir, regions, outdir)
    niche_persistence(args.gw20_niche_dir, args.gw34_niche_dir, regions, outdir, args.match_thr)
    log(f"DONE. wrote persistence_{{signaling,composition,niches}}.csv (as available) to {outdir}")


if __name__ == "__main__":
    main()
