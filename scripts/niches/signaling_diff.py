#!/usr/bin/env python
"""
Reproducible V1-vs-V2 signaling difference test -- the computation the "V1↑WNT/FGF,
V2↑ECM/guidance" claim was originally eyeballed from but that no committed script
actually produced (compare_niches.py's sig() helper is defined-but-unused).

Built to settle the WNT sign flip: an early domain-level read said V1↑WNT, while the
direct-SP-cell view (sp_niche_analysis.py -> sp_cell_direct_summary.csv) found WNT
marginally V2>V1. Those used DIFFERENT cell populations, so they aren't actually in
conflict. This script computes the V1-vs-V2 per-pathway difference on explicitly-
defined, identical cell populations, keeps sender (s-) and receiver (r-) SEPARATE,
and adds a proper significance test + effect size:

  - population "cpmz": every cell in the COMMOT h5ad (COMMOT was run on CP/MZ layers,
    so this IS the CP/MZ population the domain-level claim implicitly averaged over).
  - population "sp": layer=="sp" only (the direct-view population).

V1 and V2 are physically separate dissection pieces, so their cells are independent
samples -> Mann-Whitney U per pathway/direction, BH-FDR across pathways. At ~50k/42k
cells almost any difference is "significant", so the q-value gates and the EFFECT SIZE
(rank-biserial, log2 fold-change of means) ranks -- read effect size first.

Sign convention: diff = v1_mean - v2_mean (positive = higher in V1), matching
sp_niche_analysis.py. rank_biserial > 0 likewise means V1 tends to exceed V2.

Runs in the neuro env (needs commot for CellChatDB pathway classification, + scipy).
Reuses select_signaling_cols + bh_fdr from the neurospatial package.
"""
import argparse
import time
from pathlib import Path

from neurospatial.signaling import select_signaling_cols, bh_fdr


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Reproducible V1-vs-V2 per-pathway signaling diff + significance")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--commot-dir", default="", help="dir holding commot_{region}.h5ad; default {results}/commot")
    p.add_argument("--regions", default="v1,v2", help="exactly two: region0,region1 (diff = r0_mean - r1_mean)")
    p.add_argument("--level", default="pathway", choices=["pathway", "pair", "all"])
    p.add_argument("--species", default="human")
    p.add_argument("--layer-key", default="layer")
    p.add_argument("--sp-layer", default="sp")
    p.add_argument("--eps", type=float, default=1e-9, help="pseudocount for log2 fold-change")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import commot as ct
    from scipy.stats import mannwhitneyu

    results = Path(args.results)
    commot_dir = Path(args.commot_dir) if args.commot_dir else results / "commot"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = args.regions.split(",")
    assert len(regions) == 2, "need exactly two regions"
    r0, r1 = regions

    db = ct.pp.ligand_receptor_database(species=args.species, signaling_type=None, database="CellChat")
    pathway_names = set(db.iloc[:, 2].astype(str))
    pair_suffixes = {f"{r.iloc[0]}-{r.iloc[1]}" for _, r in db.iterrows()}

    # ---- load each region's signaling frame + population masks ----
    region_sig = {}   # region -> DataFrame (cells x signaling cols)
    region_masks = {}  # region -> {"cpmz": all-True, "sp": layer==sp}
    for region in regions:
        h5ad_path = commot_dir / f"commot_{region}.h5ad"
        log(f"Loading {h5ad_path}")
        a = ad.read_h5ad(h5ad_path)
        layer = a.obs[args.layer_key].astype(str).values
        sender = a.obsm["commot-cellchat-sum-sender"]
        receiver = a.obsm["commot-cellchat-sum-receiver"]
        scols = select_signaling_cols(sender.columns, args.level, pathway_names, pair_suffixes)
        rcols = select_signaling_cols(receiver.columns, args.level, pathway_names, pair_suffixes)
        sig = pd.concat([sender[scols], receiver[rcols]], axis=1)
        sig.index = a.obs.index
        region_sig[region] = sig
        region_masks[region] = {
            "cpmz": np.ones(a.n_obs, dtype=bool),
            "sp": (layer == args.sp_layer),
        }
        log(f"  {region}: {a.n_obs} cells (cpmz), {int((layer == args.sp_layer).sum())} SP cells, "
            f"{sig.shape[1]} signaling cols")

    # shared signaling columns across both regions
    common_cols = region_sig[r0].columns.intersection(region_sig[r1].columns)
    log(f"{len(common_cols)} signaling columns common to {r0} and {r1}")

    # ---- per population: per-pathway Mann-Whitney + effect sizes ----
    all_summaries = {}
    for pop in ("cpmz", "sp"):
        m0 = region_masks[r0][pop]
        m1 = region_masks[r1][pop]
        X0 = region_sig[r0].loc[:, common_cols].values[m0]   # (n0 x P)
        X1 = region_sig[r1].loc[:, common_cols].values[m1]   # (n1 x P)
        n0, n1 = X0.shape[0], X1.shape[0]
        if n0 == 0 or n1 == 0:
            log(f"[{pop}] SKIP: {r0} n={n0}, {r1} n={n1} (population empty in a region — "
                f"e.g. no subplate cells at GW34)")
            continue
        log(f"[{pop}] {r0} n={n0}  {r1} n={n1}  over {len(common_cols)} pathways")
        rows = []
        for j, col in enumerate(common_cols):
            x0, x1 = X0[:, j], X1[:, j]
            m0v, m1v = float(x0.mean()), float(x1.mean())
            diff = m0v - m1v
            log2fc = float(np.log2((m0v + args.eps) / (m1v + args.eps)))
            # both all-zero (or identical) -> no test; MWU would error/degenerate
            if (x0.max() == x0.min()) and (x1.max() == x1.min()) and x0.max() == x1.max():
                u, pval, rb = np.nan, 1.0, 0.0
            else:
                u, pval = mannwhitneyu(x0, x1, alternative="two-sided")
                rb = 2.0 * (u / (n0 * n1)) - 1.0   # rank-biserial; >0 => r0 tends to exceed r1
            direction = col[:1] if col[:2] in ("s-", "r-") else ""
            pathway = col[2:] if col[:2] in ("s-", "r-") else col
            rows.append({"column": col, "pathway": pathway, "direction": direction,
                         f"{r0}_mean": m0v, f"{r1}_mean": m1v, "diff": diff,
                         "log2fc": log2fc, "rank_biserial": rb, "u_pvalue": float(pval)})
        df = pd.DataFrame(rows)
        df["q_value"] = bh_fdr(df["u_pvalue"].values)
        df["abs_rank_biserial"] = df["rank_biserial"].abs()
        df = df.sort_values("abs_rank_biserial", ascending=False)
        out = outdir / f"signaling_diff_{pop}.csv"
        df.to_csv(out, index=False)
        all_summaries[pop] = df
        n_sig = int((df["q_value"] < 0.05).sum())
        log(f"  wrote {out.name} ({len(df)} pathway-directions, {n_sig} with q<0.05)")

    # ---- focused WNT recheck across both populations ----
    wnt_rows = []
    for pop, df in all_summaries.items():
        w = df[df["pathway"] == "WNT"]
        for _, row in w.iterrows():
            higher = r0.upper() if row["diff"] > 0 else r1.upper()
            wnt_rows.append({
                "population": pop, "column": row["column"], "direction": row["direction"],
                f"{r0}_mean": round(row[f"{r0}_mean"], 6), f"{r1}_mean": round(row[f"{r1}_mean"], 6),
                "diff": round(row["diff"], 6), "higher_in": higher,
                "log2fc": round(row["log2fc"], 4), "rank_biserial": round(row["rank_biserial"], 4),
                "u_pvalue": row["u_pvalue"], "q_value": round(row["q_value"], 6),
            })
    wnt = pd.DataFrame(wnt_rows)
    wnt.to_csv(outdir / "wnt_recheck_summary.csv", index=False)
    log("WNT recheck (both populations, sender/receiver separate):\n" + wnt.to_string(index=False))
    log(f"DONE. wrote signaling_diff_cpmz.csv, signaling_diff_sp.csv, wnt_recheck_summary.csv to {outdir}")


if __name__ == "__main__":
    main()
