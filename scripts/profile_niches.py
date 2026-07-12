#!/usr/bin/env python
"""
Niche profiling + boundary hub for the V1-vs-V2 COMMOT comparison.

Given ONE region's ENVI-COMMOT result (commot_<region>.h5ad, the fixed per-cell
signaling substrate shared by all four pipelines) plus a domain-label vector and
a cell-type-annotation vector, emit per-niche profiles that every downstream
comparison plugs into:

  - composition  : cell-type fractions per domain (from the annotation axis)
  - signaling    : mean sender/receiver score per pathway (or pair) per domain
                   -- aggregate of the FIXED per-cell COMMOT scores, so it depends
                   only on the domain partition, NOT the annotation (option (i))
  - context      : layer fractions, spatial centroid, mean cortical depth, size
  - boundary     : per-cell cross-domain neighbor fraction, boundary-cell calls,
                   per-domain boundary/sharpness stats, and a domain adjacency
                   matrix (which niches interface -> boundary-adjacent category)

Domain/annotation labels are supplied as obs columns so this runs identically on
the stand-in (layer as domain, H2_annotation as type) and, later, on real
Banksy/STAGATE domains x scVI/UCE-scGPT annotations -- they just drop in.

Signaling level selection uses the merge column convention: pathway columns have
a single hyphen (s-WNT), pair columns have >=2 (s-TGFB1-TGFBR1_TGFBR2), and the
grand total is *-total-total.
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_signaling_cols(cols, level, pathway_names, pair_suffixes):
    """Split merged commot sum columns by level. `cols` are e.g. 's-WNT',
    's-TGFB1-TGFBR1_TGFBR2', 's-MHC-I', 's-total-total' (and 'r-' equivalents).
    Classify against the actual CellChatDB pathway/pair names -- NOT a hyphen
    count -- because some pathways (MHC-I, MHC-II, ...) contain hyphens."""
    out = []
    for c in map(str, cols):
        if c.endswith("-total-total"):
            continue  # grand total; recomputed downstream if needed
        suffix = c[2:] if (c.startswith("s-") or c.startswith("r-")) else c
        if suffix in pathway_names:
            is_pathway = True
        elif suffix in pair_suffixes:
            is_pathway = False
        else:
            is_pathway = "-" not in suffix   # fallback for anything unexpected
        if level == "all" or (level == "pathway" and is_pathway) or (level == "pair" and not is_pathway):
            out.append(c)
    return out


def main():
    p = argparse.ArgumentParser(description="Per-niche composition/signaling/boundary profiling")
    p.add_argument("--h5ad", required=True, help="commot_<region>.h5ad (per-cell signaling substrate)")
    p.add_argument("--region", default="", help="tag for output filenames; default inferred from --area-key")
    p.add_argument("--domain-key", default="layer", help="column of domain/niche labels (in obs, or in --domain-parquet)")
    p.add_argument("--celltype-key", default="H2_annotation", help="obs column of cell-type labels (if no --annotation-parquet)")
    p.add_argument("--domain-parquet", default="", help="external per-cell domain table (e.g. banksy); --domain-key selects the column; attached to the commot cells by cell id.")
    p.add_argument("--annotation-parquet", default="", help="external per-cell annotation table (e.g. scvi); attached by cell id.")
    p.add_argument("--annotation-col", default="annotation", help="column in --annotation-parquet holding the cell-type label")
    p.add_argument("--layer-key", default="layer", help="obs column of cortical layer (spatial context)")
    p.add_argument("--area-key", default="area", help="obs column used to infer --region if unset")
    p.add_argument("--depth-keys", default="cortical_depth,relative_height",
                   help="comma-sep numeric obs cols to average per domain (skipped if absent)")
    p.add_argument("--level", default="pathway", choices=["pathway", "pair", "all"],
                   help="which signaling columns to aggregate")
    p.add_argument("--agg", default="mean", choices=["mean", "median"], help="signaling aggregation")
    p.add_argument("--species", default="human", help="CellChatDB species (for pathway/pair classification)")
    p.add_argument("--knn", type=int, default=15, help="spatial neighbors for the boundary graph")
    p.add_argument("--boundary-frac", type=float, default=0.5,
                   help="a cell is a boundary cell if its cross-domain neighbor fraction >= this")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import commot as ct
    from sklearn.neighbors import NearestNeighbors

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {args.h5ad}")
    a = ad.read_h5ad(args.h5ad)
    obs = a.obs
    region = args.region or (str(obs[args.area_key].iloc[0]) if args.area_key in obs else "region")
    log(f"  {a.n_obs} cells; region tag = {region}")

    if args.layer_key not in obs.columns:
        raise SystemExit(f"obs column '{args.layer_key}' not found; have {list(obs.columns)}")

    def attach(parquet, col, what):
        """Pull a per-cell label column from an external parquet, aligned to the
        commot cells by cell id (index). Errors if any cell is unlabeled."""
        df = pd.read_parquet(parquet)
        if col not in df.columns:
            raise SystemExit(f"'{col}' not in {parquet}; have {list(df.columns)}")
        s = df[col].reindex(obs.index)
        if s.isna().any():
            raise SystemExit(f"{s.isna().sum()} of {len(s)} commot cells missing from {what} parquet")
        return s.astype(str)

    if args.domain_parquet:
        domain = attach(args.domain_parquet, args.domain_key, "domain")
        log(f"  domains from {Path(args.domain_parquet).name}['{args.domain_key}']")
    else:
        domain = obs[args.domain_key].astype(str)
    if args.annotation_parquet:
        celltype = attach(args.annotation_parquet, args.annotation_col, "annotation")
        log(f"  cell types from {Path(args.annotation_parquet).name}['{args.annotation_col}']")
    else:
        celltype = obs[args.celltype_key].astype(str)
    layer = obs[args.layer_key].astype(str)
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    domains = sorted(domain.unique())
    log(f"  {len(domains)} domains ('{args.domain_key}'); {celltype.nunique()} cell types")

    # --- signaling matrix (sender + receiver, chosen level) ---
    db = ct.pp.ligand_receptor_database(species=args.species, signaling_type=None, database="CellChat")
    pathway_names = set(db.iloc[:, 2].astype(str))
    pair_suffixes = {f"{r.iloc[0]}-{r.iloc[1]}" for _, r in db.iterrows()}
    sender = a.obsm["commot-cellchat-sum-sender"]
    receiver = a.obsm["commot-cellchat-sum-receiver"]
    scols = select_signaling_cols(sender.columns, args.level, pathway_names, pair_suffixes)
    rcols = select_signaling_cols(receiver.columns, args.level, pathway_names, pair_suffixes)
    sig = pd.concat([sender[scols], receiver[rcols]], axis=1)
    sig.index = obs.index
    log(f"  signaling: {len(scols)} sender + {len(rcols)} receiver {args.level}-level columns")

    # --- composition: cell-type fractions per domain ---
    comp = pd.crosstab(domain, celltype)
    comp_frac = comp.div(comp.sum(axis=1), axis=0)
    comp_frac.index.name = "domain"
    comp_frac.to_csv(outdir / f"{region}_niche_composition.csv")

    # --- signaling profile per domain ---
    grp = sig.groupby(domain.values)
    sig_prof = grp.median() if args.agg == "median" else grp.mean()
    sig_prof.index.name = "domain"
    sig_prof.to_csv(outdir / f"{region}_niche_signaling.csv")

    # --- boundary graph: kNN, cross-domain neighbor fraction per cell ---
    log(f"  building kNN (k={args.knn}) boundary graph on spatial coords (um)")
    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]                                   # drop self
    dom_codes = domain.astype("category").cat.codes.to_numpy()
    neigh_codes = dom_codes[idx]                       # (n_cells, knn)
    cross_frac = (neigh_codes != dom_codes[:, None]).mean(axis=1)
    is_boundary = cross_frac >= args.boundary_frac

    # most common foreign domain per cell (for adjacency / labeling)
    cats = domain.astype("category").cat.categories
    foreign = np.where(neigh_codes != dom_codes[:, None], neigh_codes, -1)
    top_foreign = []
    for row in foreign:
        vals = row[row >= 0]
        top_foreign.append(cats[np.bincount(vals).argmax()] if vals.size else "")
    bcells = pd.DataFrame({"domain": domain.values, "cross_domain_frac": cross_frac,
                           "is_boundary": is_boundary, "top_foreign_domain": top_foreign},
                          index=obs.index)
    bcells.to_parquet(outdir / f"{region}_boundary_cells.parquet")

    # --- domain adjacency (interface strength between niches) ---
    adj = np.zeros((len(cats), len(cats)), dtype=float)
    for own, row in zip(dom_codes, neigh_codes):
        for nb in row:
            if nb != own:
                adj[own, nb] += 1
    adj = (adj + adj.T) / 2.0                           # symmetrize edge counts
    adj_df = pd.DataFrame(adj, index=cats, columns=cats)
    adj_df.to_csv(outdir / f"{region}_domain_adjacency.csv")

    # --- per-domain summary (the hub table) ---
    depth_keys = [k for k in args.depth_keys.split(",") if k and k in obs.columns]
    layer_frac = pd.crosstab(domain, layer)
    layer_frac = layer_frac.div(layer_frac.sum(axis=1), axis=0).add_prefix("layer_")
    rows = []
    for d in domains:
        m = (domain == d).values
        row = {"domain": d, "region": region, "n_cells": int(m.sum()),
               "centroid_x": float(xy[m, 0].mean()), "centroid_y": float(xy[m, 1].mean()),
               "boundary_cell_frac": float(is_boundary[m].mean()),
               "mean_cross_domain_frac": float(cross_frac[m].mean())}
        for dk in depth_keys:
            row[f"mean_{dk}"] = float(pd.to_numeric(obs[dk][m], errors="coerce").mean())
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("domain").join(layer_frac)
    summary.to_csv(outdir / f"{region}_niche_profile.csv")

    log(f"DONE. wrote {region}_niche_profile / _composition / _signaling / "
        f"_boundary_cells / _domain_adjacency to {outdir}")
    log(f"  domains: {len(domains)} | boundary cells: {int(is_boundary.sum())} "
        f"({100*is_boundary.mean():.1f}%) | signaling cols: {sig_prof.shape[1]}")


if __name__ == "__main__":
    main()
