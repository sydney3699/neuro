#!/usr/bin/env python
"""
Merge pair-parallel COMMOT chunks for one region into a single result.

Each chunk job (run_commot.py --pair-chunk i/N) wrote slim per-pair
sender/receiver parquets over ALL cells. This concatenates the per-pair columns
across chunks, recomputes the total and the per-pathway sums (deferred from the
chunks since a pathway's pairs are split across chunks), attaches obs + spatial,
and writes commot_<region>.h5ad — the per-cell signaling niche used downstream.
"""
import argparse
import glob
from pathlib import Path


REGION_AREA = {"v1": "A-V1", "v2": "B-V2"}
CPMZ = ["mz", "l2", "l3", "l4", "l5", "l6", "sp"]


def main():
    p = argparse.ArgumentParser(description="Merge pair-parallel COMMOT chunks")
    p.add_argument("--region", required=True, choices=["v1", "v2"])
    p.add_argument("--chunk-dir", default="/scratch/cole.sy/neuro/results/commot")
    p.add_argument("--outdir", default="/scratch/cole.sy/neuro/results/commot")
    p.add_argument("--h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad",
                   help="source imputation h5ad (for obs + spatial of the same cells).")
    p.add_argument("--area-key", default="area")
    p.add_argument("--layer-key", default="layer")
    p.add_argument("--layers", default=",".join(CPMZ))
    p.add_argument("--species", default="human")
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import commot as ct

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- pair -> pathway map (build expected column names from the DB; no fragile parsing) ---
    db = ct.pp.ligand_receptor_database(species=args.species, signaling_type=None, database="CellChat")
    col2path = {}
    for _, r in db.iterrows():
        lig, rec, pathway = str(r.iloc[0]), str(r.iloc[1]), str(r.iloc[2])
        col2path[f"s-{lig}-{rec}"] = pathway
        col2path[f"r-{lig}-{rec}"] = pathway

    def load_side(side):
        files = sorted(glob.glob(str(Path(args.chunk_dir) / f"chunk_{args.region}_*_{side}.parquet")))
        if not files:
            raise SystemExit(f"no {side} chunks for region {args.region} in {args.chunk_dir}")
        parts = [pd.read_parquet(f) for f in files]
        df = pd.concat(parts, axis=1)              # concat pair columns; same cell index
        df = df.loc[:, ~df.columns.duplicated()]   # guard against accidental overlap
        print(f"  {side}: {len(files)} chunks -> {df.shape[1]} pair columns, {df.shape[0]} cells", flush=True)
        return df

    sender = load_side("sender")
    receiver = load_side("receiver")

    # --- recompute total + per-pathway sums from the full per-pair set ---
    def add_aggregates(df, prefix):
        pair_cols = list(df.columns)
        df[f"{prefix}-total-total"] = df[pair_cols].sum(axis=1)
        paths = {}
        for c in pair_cols:
            pw = col2path.get(str(c))
            if pw:
                paths.setdefault(f"{prefix}-{pw}", []).append(c)
        for name, cols in paths.items():
            df[name] = df[cols].sum(axis=1)
        print(f"  {prefix}: {len(pair_cols)} pairs, {len(paths)} pathways", flush=True)
        return df

    sender = add_aggregates(sender, "s")
    receiver = add_aggregates(receiver, "r")

    # --- attach obs + spatial for the same region/layer cells ---
    src = ad.read_h5ad(args.h5ad, backed="r")
    area = REGION_AREA[args.region]
    mask = src.obs[args.area_key].astype(str) == area
    if args.layers != "all":
        layers = [s.strip() for s in args.layers.split(",")]
        mask &= src.obs[args.layer_key].astype(str).isin(layers)
    obs = src.obs[mask.values].copy()
    xy = np.asarray(src.obsm["spatial"])[mask.values]

    # align everything to the obs cell order
    sender = sender.reindex(obs.index)
    receiver = receiver.reindex(obs.index)
    assert not sender.isna().any().any(), "sender has cells missing from chunks"

    out = ad.AnnData(obs=obs, var=pd.DataFrame(index=["_"]),
                     X=np.zeros((obs.shape[0], 1), dtype=np.float32))
    out.obsm["spatial"] = xy
    out.obsm["commot-cellchat-sum-sender"] = sender
    out.obsm["commot-cellchat-sum-receiver"] = receiver
    out.uns["commot_merge_info"] = {"region": args.region, "n_pairs": int(sender.filter(regex=r"^s-.+-.+").shape[1])}

    dst = outdir / f"commot_{args.region}.h5ad"
    out.write_h5ad(dst)
    print(f"DONE. wrote {dst}  ({out.n_obs} cells)", flush=True)


if __name__ == "__main__":
    main()
