#!/usr/bin/env python
"""
STAGATE spatial-domain identification for the niche pipeline comparison, run
PER REGION on the ENVI-imputed expression, sweeping the domain count k.

STAGATE (Dong & Zhang, Nat Commun 2022) is a graph-attention autoencoder: it
builds a spatial neighbour graph and learns a per-cell embedding that fuses
expression with spatial context. We use the official STAGATE_pyG, then KMeans to
an EXACT k on the learned embedding -- the SAME clustering applied to the BANKSY
embedding -- so the only thing differing between the two spatial-domain arms is
the embedding, isolating the method for the comparison. Sweeps k=6..14.

Spatial graph uses k_cutoff=18 (KNN) to match BANKSY's num_neighbours=18, so both
methods see the same neighbourhood size. Runs in the isolated `stagate` conda env
(GPU). Output mirrors banksy_domains.py: per-cell domain parquet (domain_k6..k14,
keyed by cell id, consumable by profile_niches.py), per-k domain x layer cross-
tabs, and a summary CSV (spatial coherence + layer purity).
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


REGION_AREA = {"v1": "A-V1", "v2": "B-V2"}
CPMZ = ["mz", "l2", "l3", "l4", "l5", "l6", "sp"]


def main():
    p = argparse.ArgumentParser(description="STAGATE spatial domains, per region, k-sweep")
    p.add_argument("--region", required=True, choices=["v1", "v2"])
    p.add_argument("--h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad")
    p.add_argument("--area-key", default="area")
    p.add_argument("--layer-key", default="layer")
    p.add_argument("--layers", default=",".join(CPMZ))
    p.add_argument("--k-cutoff", type=int, default=18, help="KNN spatial-graph neighbours (match BANKSY).")
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    import STAGATE_pyG as st
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- region subset of the ENVI imputation ---
    log(f"Loading {args.h5ad}")
    src = ad.read_h5ad(args.h5ad)
    genes = [str(g) for g in src.uns["imputation_genes"]]
    area = REGION_AREA[args.region]
    mask = (src.obs[args.area_key].astype(str) == area).values
    if args.layers != "all":
        layers = [s.strip() for s in args.layers.split(",")]
        mask &= src.obs[args.layer_key].astype(str).isin(layers).values
    idx = np.where(mask)[0]
    X = np.asarray(src.obsm["imputation"], dtype="float32")[idx]
    xy = np.asarray(src.obsm["spatial"])[idx].astype(float)
    obs = src.obs.iloc[idx].copy()
    del src
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = xy
    log(f"  region {args.region} ({area}): {adata.n_obs} cells x {adata.n_vars} genes")

    # --- preprocess (STAGATE reconstructs log-normalized expression; no scaling) ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- spatial graph + graph-attention autoencoder embedding ---
    st.Cal_Spatial_Net(adata, k_cutoff=args.k_cutoff, model="KNN", verbose=False)
    st.Stats_Spatial_Net(adata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  train_STAGATE on {device} (k_cutoff={args.k_cutoff}, n_epochs={args.n_epochs})")
    adata = st.train_STAGATE(adata, n_epochs=args.n_epochs, random_seed=args.seed, device=device)
    emb = np.asarray(adata.obsm["STAGATE"])
    log(f"  STAGATE embedding {emb.shape}")

    # --- spatial kNN for coherence ---
    nn = NearestNeighbors(n_neighbors=args.k_cutoff + 1).fit(xy)
    _, sp_idx = nn.kneighbors(xy)
    sp_idx = sp_idx[:, 1:]
    layer = adata.obs[args.layer_key].astype(str)

    # --- KMeans exact-k sweep (identical to banksy_domains.py) ---
    labels_out = {}
    rows = []
    for k in range(args.kmin, args.kmax + 1):
        lab = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(emb)
        labels_out[f"domain_k{k}"] = [f"d{v}" for v in lab]
        coherence = float((lab[sp_idx] == lab[:, None]).mean())
        ct = pd.crosstab(pd.Series(lab, name="domain"), layer.values)
        ct.to_csv(outdir / f"{args.region}_domain_x_layer_k{k}.csv")
        frac = ct.div(ct.sum(axis=1), axis=0)
        purity = float((frac.max(axis=1) * ct.sum(axis=1) / ct.values.sum()).sum())
        rows.append({"k": k, "n_domains": int(len(set(lab))),
                     "spatial_coherence": round(coherence, 4), "layer_purity": round(purity, 4)})
        log(f"  k={k}: coherence={coherence:.3f} layer_purity={purity:.3f}")

    per_cell = pd.DataFrame(labels_out, index=adata.obs_names)
    per_cell.to_parquet(outdir / f"{args.region}_stagate_domains.parquet")
    pd.DataFrame(rows).to_csv(outdir / f"{args.region}_stagate_ksweep_summary.csv", index=False)
    log(f"DONE {args.region}. wrote {args.region}_stagate_domains.parquet (k={args.kmin}..{args.kmax}) "
        f"+ ksweep summary + per-k crosstabs to {outdir}")
    log("  summary:\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
