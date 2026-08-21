#!/usr/bin/env python
"""
Embedding-agnostic cell annotation scaffold for the V1-vs-V2 pipeline comparison.

  embed -> Leiden cluster -> cluster-to-reference-centroid annotation

The ONLY thing that varies across annotation pipelines is the embedding backend
(scVI = standard, scGPT/UCE = foundation-model); Leiden and the reference-centroid
label transfer are held IDENTICAL, so composition differences are attributable to
the embedding alone (methodological Q1). Backends:

  --embedding scvi   : scVI latent on ENVI-imputed expression. scVI's likelihood
                       is a count model, but ENVI imputation is continuous, so we
                       round to integer PSEUDO-COUNTS and use NB. (Documented
                       deviation; flag if you'd prefer PCA-on-log or raw-panel scVI.)
  --embedding pca    : PCA on log-normalized imputed expression. No GPU/torch;
                       validates the scaffold end-to-end and serves as a baseline.
  --embedding-obsm K : skip computing an embedding, read a precomputed one from the
                       spatial h5ad's obsm[K] (this is how scGPT/UCE embeddings,
                       computed in their own envs, drop into the same scaffold).

Expression = the ENVI imputation (obsm['imputation'] / uns['imputation_genes']) --
the baseline-preprocessed substrate used everywhere in the project. Reference =
snRNA-seq with expert 'celltypes'. Output is a per-cell label table keyed by cell
id, consumable by profile_niches.py as the annotation axis.
"""
import argparse
import time
from pathlib import Path

from neurospatial.confidence import compute_low_confidence


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_embedding(adata, backend, n_latent, scvi_epochs, seed, log):
    """Return an (n_cells, d) latent array in adata.obsm['X_emb']-ready form."""
    import scanpy as sc
    if backend == "pca":
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.scale(a, max_value=10)
        sc.pp.pca(a, n_comps=n_latent)
        return a.obsm["X_pca"]
    if backend == "scvi":
        import numpy as np
        import scvi
        scvi.settings.seed = seed          # reproducible latent (scVI training is stochastic)
        a = adata.copy()
        # ENVI imputation is continuous; scVI needs counts -> integer pseudo-counts.
        a.layers["counts"] = np.rint(np.asarray(a.X)).clip(min=0).astype("float32")
        scvi.model.SCVI.setup_anndata(a, layer="counts")
        model = scvi.model.SCVI(a, n_latent=n_latent)
        model.train(max_epochs=scvi_epochs)
        return model.get_latent_representation()
    raise SystemExit(f"unknown embedding backend {backend}")


def main():
    p = argparse.ArgumentParser(description="Embed -> Leiden -> reference-centroid annotation")
    p.add_argument("--spatial-h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad")
    p.add_argument("--reference-h5ad", default="/scratch/cole.sy/neuro/data/raw/snrna.h5ad")
    p.add_argument("--ref-label-key", default="celltypes", help="reference obs column with cell-type labels")
    p.add_argument("--embedding", default="scvi", choices=["scvi", "pca"],
                   help="embedding backend (ignored if --embedding-obsm is set)")
    p.add_argument("--embedding-obsm", default="", help="read a precomputed embedding from spatial obsm[K] (e.g. scGPT/UCE)")
    p.add_argument("--embedding-parquet", default="", help="read a precomputed per-cell embedding from a parquet keyed by cell id (e.g. scGPT); attached by index")
    p.add_argument("--emb-name", default="", help="label for output filenames when using --embedding-parquet (e.g. scgpt)")
    p.add_argument("--leiden-resolution", type=float, default=1.0)
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--scvi-epochs", type=int, default=200)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--seed", type=int, default=0, help="reproducibility seed (scvi/leiden/kmeans)")
    p.add_argument("--min-corr", type=float, default=0.3,
                   help="absolute floor: flag a cluster low_confidence if its best reference correlation is below this")
    p.add_argument("--margin-percentile", type=float, default=25.0,
                   help="flag a cluster if its best-minus-second-best margin is at/below this within-run "
                        "percentile AND the 3 metrics (Pearson/Spearman/cosine) don't all agree on the pick")
    p.add_argument("--min-margin", type=float, default=None,
                   help="optional absolute margin floor (old fixed-0.05 behavior); disabled by default")
    p.add_argument("--unassign", action="store_true",
                   help="set low_confidence clusters to 'unassigned' rather than only flagging them")
    p.add_argument("--tag", default="FB080", help="output filename tag")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backend = ("parquet:" + (args.emb_name or Path(args.embedding_parquet).stem) if args.embedding_parquet
               else "obsm:" + args.embedding_obsm if args.embedding_obsm else args.embedding)

    # --- spatial cells: ENVI-imputed expression as an AnnData ---
    log(f"Loading spatial imputation {args.spatial_h5ad}")
    src = ad.read_h5ad(args.spatial_h5ad)
    genes = [str(g) for g in src.uns["imputation_genes"]]
    adata = ad.AnnData(X=np.asarray(src.obsm["imputation"], dtype="float32"),
                       obs=src.obs.copy(), var=pd.DataFrame(index=genes))
    if "spatial" in src.obsm:
        adata.obsm["spatial"] = np.asarray(src.obsm["spatial"])
    del src
    log(f"  {adata.n_obs} cells x {adata.n_vars} imputation genes")

    # --- embedding (external parquet > obsm > compute) ---
    emb_label = args.emb_name or args.embedding_obsm or args.embedding
    if args.embedding_parquet:
        edf = pd.read_parquet(args.embedding_parquet).reindex(adata.obs_names)
        if edf.isna().any().any():
            raise SystemExit(f"{int(edf.isna().any(axis=1).sum())} cells missing from {args.embedding_parquet}")
        emb = edf.to_numpy(dtype="float32")
        emb_label = args.emb_name or Path(args.embedding_parquet).stem
        log(f"  using precomputed embedding {Path(args.embedding_parquet).name} {emb.shape}")
    elif args.embedding_obsm:
        if args.embedding_obsm not in adata.obsm:
            raise SystemExit(f"obsm['{args.embedding_obsm}'] not found for precomputed embedding")
        emb = np.asarray(adata.obsm[args.embedding_obsm])
        log(f"  using precomputed embedding obsm['{args.embedding_obsm}'] {emb.shape}")
    else:
        log(f"  computing {args.embedding} embedding (n_latent={args.n_latent})")
        emb = compute_embedding(adata, args.embedding, args.n_latent, args.scvi_epochs, args.seed, log)
    adata.obsm["X_emb"] = emb

    # --- Leiden on the embedding (shared across backends) ---
    log(f"  Leiden (res={args.leiden_resolution}) on embedding")
    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=args.n_neighbors)
    sc.tl.leiden(adata, resolution=args.leiden_resolution, key_added="leiden", flavor="igraph",
                 n_iterations=2, random_state=args.seed)
    n_clusters = adata.obs["leiden"].nunique()
    log(f"  {n_clusters} Leiden clusters")

    # --- reference centroids over shared genes, log-normalized both sides ---
    log(f"Loading reference {args.reference_h5ad}")
    ref = ad.read_h5ad(args.reference_h5ad)
    if args.ref_label_key not in ref.obs:
        raise SystemExit(f"reference has no obs['{args.ref_label_key}']; have {list(ref.obs.columns)}")
    shared = [g for g in genes if g in set(map(str, ref.var_names))]
    log(f"  {len(shared)} shared genes (of {len(genes)} imputation genes)")

    def lognorm_mean_by(a_expr, var_names, groups, shared_genes):
        """Return DataFrame [group x shared_genes] of log1p-CP10k mean profiles."""
        sub = ad.AnnData(X=a_expr, var=pd.DataFrame(index=[str(v) for v in var_names]))
        sub = sub[:, shared_genes].copy()
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
        df = pd.DataFrame(X, columns=shared_genes)
        df["_g"] = np.asarray(groups)
        return df.groupby("_g").mean()

    ref_cent = lognorm_mean_by(ref.X, ref.var_names, ref.obs[args.ref_label_key].astype(str).values, shared)
    clus_mean = lognorm_mean_by(adata.X, adata.var_names, adata.obs["leiden"].astype(str).values, shared)
    del ref

    # --- cluster -> reference centroid, with confidence + multi-metric consensus ---
    # Primary transfer = Pearson argmax (kept as the single reproducible mechanism so
    # both arms are directly comparable). But we ALSO record: the best-vs-second-best
    # margin (forcing a label when top two are near-tied is unreliable -- common for
    # transitional/gradient states in developing cortex), and whether Spearman & cosine
    # agree with the Pearson pick (scmap-style consensus). Low-confidence clusters are
    # flagged (and optionally set to 'unassigned' with --unassign) for manual review
    # rather than silently trusted.
    from scipy.stats import rankdata
    from sklearn.metrics.pairwise import cosine_similarity
    C, T = clus_mean.values, ref_cent.values
    nC = C.shape[0]
    pear = np.corrcoef(np.vstack([C, T]))[:nC, nC:]                       # clusters x types
    spear = np.corrcoef(np.vstack([rankdata(C, axis=1), rankdata(T, axis=1)]))[:nC, nC:]
    cos = cosine_similarity(C, T)
    order = np.argsort(-pear, axis=1)
    best, second = order[:, 0], order[:, 1]
    ridx = np.arange(nC)
    best_corr = pear[ridx, best]
    margin = best_corr - pear[ridx, second]
    consensus = 1 + (spear.argmax(1) == best).astype(int) + (cos.argmax(1) == best).astype(int)
    low_conf = compute_low_confidence(best_corr, margin, consensus,
                                      min_corr=args.min_corr,
                                      margin_percentile=args.margin_percentile,
                                      min_margin_floor=args.min_margin)
    labels = ref_cent.index.to_numpy()[best].astype(object)
    if args.unassign:
        labels[low_conf] = "unassigned"
    cluster_map = pd.DataFrame({
        "leiden": clus_mean.index,
        "annotation": labels,
        "best_corr": best_corr.round(4),
        "second_label": ref_cent.index.to_numpy()[second],
        "second_corr": pear[ridx, second].round(4),
        "margin": margin.round(4),
        "spearman_label": ref_cent.index.to_numpy()[spear.argmax(1)],
        "cosine_label": ref_cent.index.to_numpy()[cos.argmax(1)],
        "metric_consensus": consensus,                                   # 1..3 metrics agreeing w/ Pearson
        "low_confidence": low_conf,
    }).set_index("leiden")
    nlc = int(low_conf.sum())
    log(f"  cluster->ref: {nlc}/{nC} low-confidence "
        f"(corr<{args.min_corr}, or margin<=p{args.margin_percentile:g} AND consensus<3"
        f"{'' if args.min_margin is None else f', or margin<{args.min_margin}'}); "
        f"full 3-metric consensus on {int((consensus == 3).sum())}/{nC}")
    cluster_map.to_csv(outdir / f"{args.tag}_{emb_label}_cluster_map.csv")

    # --- per-cell label table (keyed by cell id; consumed by profile_niches) ---
    lab = adata.obs["leiden"].astype(str).map(cluster_map["annotation"])
    per_cell = pd.DataFrame({"leiden": adata.obs["leiden"].astype(str).values,
                             "annotation": lab.values}, index=adata.obs_names)
    per_cell.to_parquet(outdir / f"{args.tag}_{emb_label}_annotation.parquet")

    log(f"DONE ({backend}). {n_clusters} clusters -> {cluster_map['annotation'].nunique()} "
        f"reference types; median cluster-centroid corr = {cluster_map['best_corr'].median():.3f}")
    log(f"  wrote {args.tag}_{emb_label}_annotation.parquet + _cluster_map.csv to {outdir}")


if __name__ == "__main__":
    main()
