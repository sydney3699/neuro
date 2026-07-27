#!/usr/bin/env python
"""
Subplate (SP)-focused niche analysis for the V1-vs-V2 comparison -- the
project brief's secondary signaling angle (see COMMOT design notes: at
dis_thr=150um, SP is the only CP/MZ layer with meaningful proximity to
germinal-zone cells).

Two complementary, upstream-compute-free views:

  View 1 (domain-enrichment): uses the existing per-k domain x layer
  crosstabs (banksy_domains.py/stagate_domains.py's {region}_domain_x_layer_k{k}.csv)
  to flag domains enriched for SP cells, then pulls those domains' signaling
  profiles from the corresponding niche_{r}_{m}K{k}_{a}/{r}_niche_signaling.csv.
  Tests whether an SP-enriched domain is consistent across Banksy/STAGATE and
  across k, and how its signaling differs V1 vs V2.

  View 2 (direct SP-cell view): filters COMMOT h5ad cells to layer=="sp"
  directly (no domain assignment involved), and computes a domain-independent
  per-annotation composition + region-level mean signaling-pathway profile,
  as a sanity check on View 1 and a direct V1-vs-V2 SP signaling comparison.
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_signaling_cols(cols, level, pathway_names, pair_suffixes):
    """Identical logic to profile_niches.py: classify merged commot sum
    columns (e.g. 's-WNT', 's-TGFB1-TGFBR1_TGFBR2', 's-total-total') by
    level against the actual CellChatDB pathway/pair names."""
    out = []
    for c in map(str, cols):
        if c.endswith("-total-total"):
            continue
        suffix = c[2:] if (c.startswith("s-") or c.startswith("r-")) else c
        if suffix in pathway_names:
            is_pathway = True
        elif suffix in pair_suffixes:
            is_pathway = False
        else:
            is_pathway = "-" not in suffix
        if level == "all" or (level == "pathway" and is_pathway) or (level == "pair" and not is_pathway):
            out.append(c)
    return out


def main():
    p = argparse.ArgumentParser(description="Subplate (SP)-focused niche analysis")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--annotations", default="scvi,scgpt")
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=14)
    p.add_argument("--enrich-mult", type=float, default=2.0,
                   help="a domain is SP-enriched if its SP fraction >= this x the region's overall SP frequency")
    p.add_argument("--level", default="pathway", choices=["pathway", "pair", "all"])
    p.add_argument("--species", default="human")
    p.add_argument("--top-n", type=int, default=10, help="top-N pathways/domains to report")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import commot as ct

    results = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    regions = args.regions.split(",")
    methods = args.methods.split(",")
    annotations = args.annotations.split(",")

    db = ct.pp.ligand_receptor_database(species=args.species, signaling_type=None, database="CellChat")
    pathway_names = set(db.iloc[:, 2].astype(str))
    pair_suffixes = {f"{r.iloc[0]}-{r.iloc[1]}" for _, r in db.iterrows()}

    # ---------------- per-region: load h5ad once, reused by both views ----------------
    region_data = {}
    for region in regions:
        h5ad_path = results / "commot" / f"commot_{region}.h5ad"
        log(f"Loading {h5ad_path}")
        a = ad.read_h5ad(h5ad_path)
        layer = a.obs["layer"].astype(str)
        sp_mask = (layer == "sp").values
        sp_freq = float(sp_mask.mean())
        sender = a.obsm["commot-cellchat-sum-sender"]
        receiver = a.obsm["commot-cellchat-sum-receiver"]
        scols = select_signaling_cols(sender.columns, args.level, pathway_names, pair_suffixes)
        rcols = select_signaling_cols(receiver.columns, args.level, pathway_names, pair_suffixes)
        sig = pd.concat([sender[scols], receiver[rcols]], axis=1)
        sig.index = a.obs.index
        region_data[region] = {"obs_index": a.obs.index, "sp_mask": sp_mask,
                                "sp_freq": sp_freq, "sig": sig, "n_cells": a.n_obs}
        log(f"  {region}: {a.n_obs} cells, SP frequency={sp_freq:.4f}, "
            f"{len(scols)} sender + {len(rcols)} receiver {args.level}-level cols")
        del a

    # ================= View 1: domain-enrichment =================
    v1_rows = []
    for region in regions:
        sp_freq = region_data[region]["sp_freq"]
        for method in methods:
            for k in range(args.kmin, args.kmax + 1):
                xtab_path = results / method / f"{region}_domain_x_layer_k{k}.csv"
                if not xtab_path.exists():
                    continue
                xtab = pd.read_csv(xtab_path, index_col=0)
                if "sp" not in xtab.columns:
                    continue
                frac = xtab.div(xtab.sum(axis=1), axis=0)
                sp_frac = frac["sp"]
                enriched = sp_frac[sp_frac >= args.enrich_mult * sp_freq] if sp_freq > 0 else sp_frac.iloc[0:0]
                for dom_int, sfrac in enriched.items():
                    domain_id = f"d{int(dom_int)}"
                    row = {"region": region, "method": method, "k": k,
                           "domain": domain_id, "sp_fraction": float(sfrac),
                           "region_sp_freq": sp_freq,
                           "enrichment_ratio": float(sfrac / sp_freq) if sp_freq > 0 else float("nan"),
                           "domain_n_cells": int(xtab.loc[dom_int].sum())}
                    for annot in annotations:
                        sig_path = results / f"niche_{region}_{method}K{k}_{annot}" / f"{region}_niche_signaling.csv"
                        if not sig_path.exists():
                            row[f"top_pathways_{annot}"] = ""
                            continue
                        sig_df = pd.read_csv(sig_path, index_col=0)
                        if domain_id not in sig_df.index:
                            row[f"top_pathways_{annot}"] = ""
                            continue
                        prof = sig_df.loc[domain_id]
                        top = prof.sort_values(ascending=False).head(args.top_n)
                        row[f"top_pathways_{annot}"] = "; ".join(f"{c}={v:.2f}" for c, v in top.items())
                    v1_rows.append(row)
    v1_df = pd.DataFrame(v1_rows)
    v1_df.to_csv(outdir / "sp_niche_summary.csv", index=False)
    n_k_avail = sorted(v1_df["k"].unique().tolist()) if len(v1_df) else []
    log(f"View 1: {len(v1_df)} SP-enriched (region,method,k,domain) rows across k={n_k_avail} -> sp_niche_summary.csv")

    # ================= View 2: direct SP-cell view =================
    v2_rows = []
    sig_by_region = {}
    for region in regions:
        d = region_data[region]
        sp_mask = d["sp_mask"]
        sp_ids = d["obs_index"][sp_mask]
        sig_sp_mean = d["sig"].loc[sp_ids].mean()
        sig_by_region[region] = sig_sp_mean
        for annot in annotations:
            ann_path = results / f"annot_{annot}" / f"FB080_{annot}_annotation.parquet"
            if not ann_path.exists():
                continue
            ann = pd.read_parquet(ann_path)
            ann_sp = ann["annotation"].reindex(sp_ids)
            comp = ann_sp.value_counts(normalize=True, dropna=True)
            for celltype, fr in comp.items():
                v2_rows.append({"region": region, "annotation": annot, "celltype": celltype,
                                 "fraction": float(fr), "n_sp_cells": int(sp_mask.sum())})
    v2_comp_df = pd.DataFrame(v2_rows)
    v2_comp_df.to_csv(outdir / "sp_cell_direct_composition.csv", index=False)
    log(f"View 2 composition: {len(v2_comp_df)} rows -> sp_cell_direct_composition.csv")

    if len(regions) >= 2:
        r0, r1 = regions[0], regions[1]
        s0, s1 = sig_by_region[r0], sig_by_region[r1]
        common = s0.index.intersection(s1.index)
        summary = pd.DataFrame({f"{r0}_mean": s0[common], f"{r1}_mean": s1[common]})
        summary["diff"] = summary[f"{r0}_mean"] - summary[f"{r1}_mean"]
        summary["abs_diff"] = summary["diff"].abs()
        summary = summary.sort_values("abs_diff", ascending=False)
        summary.to_csv(outdir / "sp_cell_direct_summary.csv")
        log(f"Top {args.top_n} differentiating SP-cell pathways ({r0} vs {r1}):\n" +
            summary.head(args.top_n).to_string())
    else:
        pd.DataFrame({region: sig_by_region[region] for region in regions}).to_csv(
            outdir / "sp_cell_direct_summary.csv")

    log(f"DONE. wrote sp_niche_summary.csv, sp_cell_direct_composition.csv, "
        f"sp_cell_direct_summary.csv to {outdir}")


if __name__ == "__main__":
    main()
