#!/usr/bin/env python
"""
COMMOT spatial cell-cell communication on the ENVI L-R imputation, run
PER REGION (V1 or V2 separately).

A/B area labels are dissection-piece IDs: in FB080-O1b the occipital block was
cut into Piece A (V1 region, area 'A-V1') and Piece B (V2 region, 'B-V2'),
~1.7 mm apart. The two pieces never interacted in vivo, so COMMOT is run on
each region independently and the inferred niches are compared downstream
(compare_commot.py) for the V1-vs-V2 signaling story.

Input is the L-R-inclusive imputation (results/envi_FB080_lr/, 2931 genes,
~83% CellChatDB pair coverage). Imputed values are linear-scale, non-library-
normalized expression, so we normalize_total + log1p (identically per region)
before COMMOT. Coordinates are in microns; dis_thr defaults to 150 um.
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


REGION_AREA = {"v1": "A-V1", "v2": "B-V2"}
CPMZ = ["mz", "l2", "l3", "l4", "l5", "l6", "sp"]


def main():
    p = argparse.ArgumentParser(description="Per-region COMMOT on ENVI imputation")
    p.add_argument("--region", required=True, choices=["v1", "v2"])
    p.add_argument("--h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad")
    p.add_argument("--outdir", default="/scratch/cole.sy/neuro/results/commot")
    p.add_argument("--area-key", default="area")
    p.add_argument("--layer-key", default="layer")
    p.add_argument("--layers", default=",".join(CPMZ),
                   help="comma-separated layers to keep; 'all' to skip the layer filter.")
    p.add_argument("--dis-thr", type=float, default=150.0, help="interaction distance cutoff (um).")
    p.add_argument("--species", default="human")
    p.add_argument("--signaling-type", default="none",
                   help="CellChatDB signaling_type (e.g. 'Secreted Signaling'); 'none' = all types.")
    p.add_argument("--min-cell-pct", type=float, default=0.05,
                   help="drop L-R pairs expressed in fewer than this fraction of cells.")
    p.add_argument("--normalize", default="total_log", choices=["total_log", "log", "none"])
    p.add_argument("--max-cells", type=int, default=0, help="subsample cap for smoke tests; 0 = all.")
    p.add_argument("--max-pairs", type=int, default=0, help="cap L-R pairs (profiling only); 0 = all.")
    p.add_argument("--cot-nitermax", type=int, default=10000, help="OT max iterations per pair.")
    p.add_argument("--pair-chunk", default="",
                   help="'i/N': run only pair-chunk i of N (round-robin) on ALL cells, saving a slim "
                        "per-pair sender/receiver parquet for later merge. Enables pair-parallel runs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sca
    import commot as ct

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {args.h5ad}")
    src = ad.read_h5ad(args.h5ad)
    imp = np.asarray(src.obsm["imputation"], dtype=np.float32)
    genes = list(map(str, src.uns["imputation_genes"]))
    adata = ad.AnnData(X=imp, obs=src.obs.copy(),
                       var=pd.DataFrame(index=genes),
                       obsm={"spatial": np.asarray(src.obsm["spatial"])})
    del src
    log(f"  full AnnData {adata.shape}")

    # --- Region + layer subset ---
    area = REGION_AREA[args.region]
    mask = adata.obs[args.area_key].astype(str) == area
    if args.layers != "all":
        layers = [s.strip() for s in args.layers.split(",")]
        mask &= adata.obs[args.layer_key].astype(str).isin(layers)
    adata = adata[mask.values].copy()
    log(f"  region {args.region} (area={area}, layers={args.layers}): {adata.n_obs} cells")
    if adata.n_obs == 0:
        raise SystemExit("no cells after subset — check area/layer keys")

    if args.max_cells and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        adata = adata[keep].copy()
        log(f"  subsampled to {adata.n_obs} cells (smoke test)")

    # --- Normalize (identical per region so the comparison is valid) ---
    if args.normalize == "total_log":
        sca.pp.normalize_total(adata, target_sum=1e4)
        sca.pp.log1p(adata)
    elif args.normalize == "log":
        sca.pp.log1p(adata)
    log(f"  normalization: {args.normalize}")

    # --- CellChatDB, filtered to expressed pairs in THIS region ---
    st = None if args.signaling_type == "none" else args.signaling_type
    db = ct.pp.ligand_receptor_database(species=args.species, signaling_type=st, database="CellChat")
    log(f"  CellChatDB pairs (signaling_type={st}): {len(db)}")
    db_f = ct.pp.filter_lr_database(db, adata, heteromeric=True, min_cell_pct=args.min_cell_pct)
    log(f"  pairs after expression filter (min_cell_pct={args.min_cell_pct}): {len(db_f)}")
    if args.max_pairs and len(db_f) > args.max_pairs:
        db_f = db_f.iloc[:args.max_pairs].copy()
        log(f"  capped to {len(db_f)} pairs (profiling)")

    # Pair-parallel: keep every cell, run only chunk i of N pairs (round-robin so
    # slow pairs spread evenly). pathway_sum is deferred to the merge, which has all pairs.
    chunked = bool(args.pair_chunk)
    if chunked:
        ci, cn = (int(x) for x in args.pair_chunk.split("/"))
        db_f = db_f.iloc[ci::cn].copy()
        log(f"  pair-chunk {ci}/{cn}: {len(db_f)} pairs on all {adata.n_obs} cells")

    # --- Spatial communication ---
    log(f"Running spatial_communication (dis_thr={args.dis_thr} um, nitermax={args.cot_nitermax}) ...")
    t0 = time.time()
    ct.tl.spatial_communication(adata, database_name="cellchat", df_ligrec=db_f,
                                dis_thr=args.dis_thr, heteromeric=True, pathway_sum=not chunked,
                                cot_nitermax=args.cot_nitermax)
    log(f"  done in {time.time()-t0:.1f}s; new obsm keys: "
        f"{[k for k in adata.obsm if k.startswith('commot')]}")

    if chunked:
        # Slim output: only the per-pair sender/receiver columns (drop 's-total-total'
        # and the huge cell-by-cell obsp transport matrices). Merge step reassembles.
        ci, cn = (int(x) for x in args.pair_chunk.split("/"))
        for side, key in [("sender", "commot-cellchat-sum-sender"),
                          ("receiver", "commot-cellchat-sum-receiver")]:
            df = adata.obsm[key]
            pair_cols = [c for c in df.columns if str(c) not in ("s-total-total", "r-total-total")]
            df = df[pair_cols].copy()
            df.index = adata.obs_names
            df.to_parquet(outdir / f"chunk_{args.region}_{ci}of{cn}_{side}.parquet")
        log(f"DONE (chunk {ci}/{cn}). wrote slim sender/receiver parquets to {outdir}")
        return

    # --- Full save: guard h5ad write against '/' in obsm DataFrame columns (h5py
    #     reads '/' as a group path; bit us on the niche tables). ---
    for k in list(adata.obsm):
        v = adata.obsm[k]
        if isinstance(v, pd.DataFrame):
            bad = [c for c in v.columns if "/" in str(c)]
            if bad:
                adata.obsm[k] = v.rename(columns={c: str(c).replace("/", "_") for c in bad})
                log(f"  sanitized {len(bad)} '/'-containing columns in obsm['{k}']")

    out = outdir / f"commot_{args.region}.h5ad"
    adata.write_h5ad(out)
    log(f"DONE. wrote {out}  ({adata.n_obs} cells, {len(db_f)} pairs)")


if __name__ == "__main__":
    main()
