#!/usr/bin/env python
"""
BANKSY spatial-domain identification for the niche pipeline comparison, run
PER REGION on the ENVI-imputed expression, sweeping the domain count k.

BANKSY (Singhal et al., Nat Genet 2024) augments each cell's expression with its
neighbourhood mean (m=0) and azimuthal Gabor filter (m=1), weighted by lambda;
lambda=0.8 = domain-segmentation mode. We use the official `banksy` package
(v1.3.5) to build the augmented matrix (initialize_banksy -> generate_banksy_
matrix), PCA-reduce it, then KMeans to an EXACT k -- so the same clustering is
applied to BANKSY and (later) STAGATE embeddings, isolating the embedding choice
for the methodological comparison. Sweeps k=6..14 for the primary-k decision.

Runs in the isolated `banksy` conda env. Output per region is a per-cell domain
table (parquet, columns domain_k6..domain_k14) consumable by profile_niches.py,
plus per-k domain x layer cross-tabs and a summary CSV (coherence + layer purity).
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


REGION_AREA = {"v1": "A-V1", "v2": "B-V2"}
CPMZ = ["mz", "l2", "l3", "l4", "l5", "l6", "sp"]


def main():
    p = argparse.ArgumentParser(description="BANKSY spatial domains, per region, k-sweep")
    p.add_argument("--region", required=True, choices=["v1", "v2"])
    p.add_argument("--h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad")
    p.add_argument("--area-key", default="area")
    p.add_argument("--layer-key", default="layer")
    p.add_argument("--layers", default=",".join(CPMZ), help="comma-sep layers to keep; 'all' to skip.")
    p.add_argument("--lam", type=float, default=0.8, help="BANKSY lambda (0.8 = domain segmentation).")
    p.add_argument("--num-neighbours", type=int, default=18)
    p.add_argument("--max-m", type=int, default=1)
    p.add_argument("--pca-dims", type=int, default=20)
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix

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
    adata.obs["x"] = xy[:, 0]
    adata.obs["y"] = xy[:, 1]
    log(f"  region {args.region} ({area}): {adata.n_obs} cells x {adata.n_vars} genes")

    # --- preprocess (BANKSY expects scaled log-normalized expression) ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # --- BANKSY augmented matrix at lambda (domain mode) ---
    log(f"  initialize_banksy (num_neighbours={args.num_neighbours}, max_m={args.max_m})")
    banksy_dict = initialize_banksy(
        adata, coord_keys=("x", "y", "spatial"),
        num_neighbours=args.num_neighbours, nbr_weight_decay="scaled_gaussian",
        max_m=args.max_m, plt_edge_hist=False, plt_nbr_weights=False,
        plt_agf_angles=False, plt_theta=False,
    )
    log(f"  generate_banksy_matrix (lambda={args.lam})")
    banksy_dict, banksy_matrix = generate_banksy_matrix(adata, banksy_dict, [args.lam], args.max_m)

    decay = list(banksy_dict.keys())[0]
    try:
        mat = banksy_dict[decay][args.lam]["adata"].X
    except (KeyError, TypeError):
        mat = banksy_matrix
    mat = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
    if np.iscomplexobj(mat):                      # m=1 AGF is complex -> keep real+imag
        mat = np.hstack([mat.real, mat.imag]).astype("float32")
    else:
        mat = mat.astype("float32")
    log(f"  BANKSY matrix {mat.shape}; PCA -> {args.pca_dims} dims")
    emb = PCA(n_components=args.pca_dims, random_state=args.seed).fit_transform(mat)

    # --- spatial kNN for the coherence metric ---
    nn = NearestNeighbors(n_neighbors=args.num_neighbours + 1).fit(xy)
    _, sp_idx = nn.kneighbors(xy)
    sp_idx = sp_idx[:, 1:]
    layer = adata.obs[args.layer_key].astype(str)

    # --- KMeans exact-k sweep ---
    labels_out = {}
    rows = []
    for k in range(args.kmin, args.kmax + 1):
        lab = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(emb)
        col = f"domain_k{k}"
        labels_out[col] = [f"d{v}" for v in lab]
        coherence = float((lab[sp_idx] == lab[:, None]).mean())
        ct = pd.crosstab(pd.Series(lab, name="domain"), layer.values)
        ct.to_csv(outdir / f"{args.region}_domain_x_layer_k{k}.csv")
        frac = ct.div(ct.sum(axis=1), axis=0)
        purity = float((frac.max(axis=1) * ct.sum(axis=1) / ct.values.sum()).sum())  # size-weighted
        rows.append({"k": k, "n_domains": int(len(set(lab))),
                     "spatial_coherence": round(coherence, 4),
                     "layer_purity": round(purity, 4)})
        log(f"  k={k}: coherence={coherence:.3f} layer_purity={purity:.3f}")

    per_cell = pd.DataFrame(labels_out, index=adata.obs_names)
    per_cell.to_parquet(outdir / f"{args.region}_banksy_domains.parquet")
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"{args.region}_banksy_ksweep_summary.csv", index=False)
    log(f"DONE {args.region}. wrote {args.region}_banksy_domains.parquet (k={args.kmin}..{args.kmax}) "
        f"+ ksweep summary + per-k crosstabs to {outdir}")
    log("  summary:\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
