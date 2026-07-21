#!/usr/bin/env python
"""
Zero-shot scGPT cell embedding for the FM annotation arm of the 2x2.

Embeds ALL spatial cells' ENVI-imputed expression with the pretrained scGPT
whole-human model (no fine-tuning), then writes the per-cell embedding as a
parquet keyed by cell id. That embedding feeds the SHARED annotation scaffold
(annotate_cells.py --embedding-parquet), so scGPT and scVI go through the
identical Leiden + reference-centroid transfer -- only the embedding differs.

Runs in the isolated `scgpt` conda env (torch 2.3.0+cu121; flash-attn not
required). scGPT's embed_data takes an AnnData with a gene-name column matching its vocab
and bins expression internally, so we pass the continuous ENVI-imputed values
directly (clip>=0). Do NOT round to pseudo-counts -- rounding zeroed out cells
whose imputed values are all small, causing a zero-size-array error in scGPT's
per-cell tokenization. Genes absent from the vocab are dropped by scGPT (~2378/
2931 of ours match).
"""
import argparse
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Zero-shot scGPT embedding of the spatial cells")
    p.add_argument("--spatial-h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad")
    p.add_argument("--model-dir", default="/scratch/cole.sy/neuro/models/scGPT_human")
    p.add_argument("--gene-col", default="gene_name", help="var column scGPT matches to its vocab")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--tag", default="FB080")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from scgpt.tasks import embed_data

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Loading spatial imputation {args.spatial_h5ad}")
    src = ad.read_h5ad(args.spatial_h5ad)
    genes = [str(g) for g in src.uns["imputation_genes"]]
    X = np.asarray(src.obsm["imputation"], dtype="float32").clip(min=0)   # continuous; scGPT bins internally (do NOT round -> rounding zeroed some cells -> zero-size error)
    adata = ad.AnnData(X=X, obs=src.obs.copy(), var=pd.DataFrame(index=genes))
    adata.var[args.gene_col] = genes          # scGPT matches this column to its vocab
    obs_names = adata.obs_names.to_numpy()
    del src
    log(f"  {adata.n_obs} cells x {adata.n_vars} genes; device cuda={torch.cuda.is_available()}")

    log(f"Running scGPT zero-shot embed_data (model={args.model_dir}, batch_size={args.batch_size})")
    t0 = time.time()
    emb_adata = embed_data(
        adata,
        model_dir=args.model_dir,
        gene_col=args.gene_col,
        batch_size=args.batch_size,
        return_new_adata=True,
    )
    key = "X_scGPT" if "X_scGPT" in emb_adata.obsm else list(emb_adata.obsm)[0]
    emb = np.asarray(emb_adata.obsm[key] if emb_adata.obsm else emb_adata.X)
    log(f"  embedding {emb.shape} in {time.time()-t0:.0f}s (obsm key '{key}')")

    df = pd.DataFrame(emb, index=obs_names,
                      columns=[f"scgpt_{i}" for i in range(emb.shape[1])])
    df.to_parquet(outdir / f"{args.tag}_scgpt_embedding.parquet")
    log(f"DONE. wrote {args.tag}_scgpt_embedding.parquet ({emb.shape[0]} cells x {emb.shape[1]} dims) to {outdir}")


if __name__ == "__main__":
    main()
