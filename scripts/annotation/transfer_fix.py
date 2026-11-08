#!/usr/bin/env python
"""
Two replacements for the cluster-mean-argmax label transfer, scored side by side.

THE DEFECT BEING FIXED
----------------------
annotate_cells.py assigns labels by: Leiden cluster -> mean log1p-CP10k profile ->
argmax Pearson against the reference centroids. embedding_discriminability.py showed
the failure is dilution, not representation: in the scGPT arm the cells of
EN-IT-L4-V1 / DL-1 / DL-2 never exceed 24-26% of any Leiden cluster, so no cluster
mean can have its argmax land on them -- while those same cells are near-perfectly
separable in the latent (ROC-AUC 0.977-0.9998). embedding_geometry.py then ruled out
the cheap fix: no rescaling/whitening/cosine variant lifts scGPT's local kNN purity
(best was +0.010). The disorder is real geometry, so the transfer itself must change.

THE TWO CANDIDATES
------------------
  nnls     Keep the Leiden scaffold; replace argmax with a non-negative least-squares
           fit of each cluster MEAN onto the full centroid matrix. A cluster that is
           26% L4-V1 gets a nonzero L4-V1 coefficient instead of being swallowed by
           the majority type. Output is a per-cluster MIXTURE, so a label can be
           recovered without any cluster being mostly that label.

  percell  Drop Leiden. Correlate each CELL's profile -- optionally averaged over its
           k nearest neighbours in the embedding -- against the centroids, argmax per
           cell. Removes the dilution mechanism at its source.

A DESIGN CONSEQUENCE THAT MUST NOT BE GLOSSED
----------------------------------------------
`percell` at --smooth-k 1 is EMBEDDING-INDEPENDENT: it is a pure gene-space
per-cell argmax, identical for every arm, and it is exactly the proxy label
embedding_discriminability.py already computes. It would "fix" label recovery by
making the scVI and scGPT arms produce the SAME annotation -- which does not answer
methodological Q1 (does the embedding choice change the niches?), it dissolves it.
The embedding re-enters only through the smoothing neighbourhood, so --smooth-k
sweeps a continuum from embedding-free (k=1) to cluster-like dilution (large k).
Report the whole sweep; a single k hides the tradeoff. `nnls` keeps Leiden and so
stays embedding-dependent by construction, which is why it is the candidate that
preserves Q1's design.

Both methods are scored on the two things that matter: do they recover the contested
labels in the arm that loses them, and are they stable across seeds (the baseline
transfer is not -- 10 scGPT labels flip across 10 Leiden seeds).

Outputs (per run):
  {tag}_{emb}_{method}_annotation.parquet   per-cell labels (percell; nnls = argmax
                                            of the cluster's mixture, for downstream
                                            compatibility)
  {tag}_{emb}_{method}_mixture.csv          nnls only: cluster x label weight matrix
  {tag}_{emb}_{method}_recovery.csv         labels reached, with the weight/cell
                                            support behind each

Runs in the scvi-annot env (numpy/pandas/scipy/scikit-learn + anndata).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from neurospatial.expression import lognorm_profiles


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="NNLS-deconvolution and per-cell label transfer")
    p.add_argument("--spatial-h5ad",
                   default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad")
    p.add_argument("--ref-centroids-csv",
                   default="/scratch/cole.sy/neuro/data/ref_centroids_snrna.csv")
    p.add_argument("--embedding-parquet", required=True,
                   help="per-cell embedding (used for Leiden in nnls, for smoothing in percell)")
    p.add_argument("--emb-name", required=True)
    p.add_argument("--method", required=True, choices=["nnls", "percell"])
    p.add_argument("--leiden-resolution", type=float, default=1.0, help="nnls only")
    p.add_argument("--n-neighbors", type=int, default=15, help="nnls Leiden graph")
    p.add_argument("--smooth-k", type=int, default=15,
                   help="percell: neighbours to average each cell's profile over. "
                        "1 = no smoothing = EMBEDDING-INDEPENDENT (see module docstring)")
    p.add_argument("--min-weight", type=float, default=0.05,
                   help="nnls: a label counts as recovered in a cluster when its "
                        "normalised mixture weight is at least this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="FB080")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import scanpy as sc
    from scipy.optimize import nnls
    from sklearn.neighbors import NearestNeighbors

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {args.spatial_h5ad}")
    src = ad.read_h5ad(args.spatial_h5ad)
    genes = [str(g) for g in src.uns["imputation_genes"]]
    X = np.asarray(src.obsm["imputation"], dtype="float32")
    cells = pd.Index(src.obs_names)
    del src
    log(f"  {len(cells)} cells x {len(genes)} genes")

    ref_cent = pd.read_csv(args.ref_centroids_csv, index_col=0)
    shared = [str(c) for c in ref_cent.columns]
    missing = [g for g in shared if g not in set(genes)]
    if missing:
        raise SystemExit(f"{len(missing)} centroid genes absent from the imputation, "
                         f"e.g. {missing[:5]}")
    labels = ref_cent.index.to_numpy().astype(str)
    log(f"  {len(ref_cent)} reference centroids x {len(shared)} shared genes")

    emb = pd.read_parquet(args.embedding_parquet).reindex(cells)
    if emb.isna().any().any():
        raise SystemExit(f"cells missing from {args.embedding_parquet}")
    emb = emb.to_numpy(dtype="float32")
    log(f"  embedding {emb.shape}")

    prof = lognorm_profiles(X, genes, shared)            # cells x shared genes
    del X

    recovery_rows = []

    if args.method == "nnls":
        # ---- Leiden exactly as the baseline builds it, so the ONLY change is the
        # ---- cluster -> label step being scored.
        log(f"Leiden (res={args.leiden_resolution}, seed={args.seed})")
        a = ad.AnnData(X=np.zeros((len(cells), 1), dtype="float32"),
                       obs=pd.DataFrame(index=cells))
        a.obsm["X_emb"] = emb
        sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=args.n_neighbors)
        sc.tl.leiden(a, resolution=args.leiden_resolution, key_added="leiden",
                     flavor="igraph", n_iterations=2, random_state=args.seed)
        leiden = a.obs["leiden"].astype(str)
        log(f"  {leiden.nunique()} clusters")

        clus_mean = prof.groupby(leiden.values).mean()
        del prof

        # ---- NNLS: cluster_mean ~= sum_t w_t * centroid_t, w >= 0.
        # Profiles are mean log1p-CP10k, so this is a linear mixture in log space --
        # an approximation (true mixing is linear in counts, not in log counts). It is
        # used because the whole transfer, published numbers included, is defined in
        # this space; the weights are therefore a RANKING of which reference types
        # explain a cluster, not a calibrated cell-type proportion. Read them that way.
        A = ref_cent.values.T                                    # genes x types
        log(f"NNLS deconvolution of {len(clus_mean)} cluster means onto {A.shape[1]} types")
        W = np.zeros((len(clus_mean), A.shape[1]), dtype="float64")
        for i, (_, row) in enumerate(clus_mean.iterrows()):
            w, _ = nnls(A, row.values.astype("float64"))
            W[i] = w
        Wn = W / np.where(W.sum(1, keepdims=True) == 0, np.nan, W.sum(1, keepdims=True))
        Wn = np.nan_to_num(Wn)
        mix = pd.DataFrame(Wn, index=clus_mean.index, columns=labels)
        mix.to_csv(outdir / f"{args.tag}_{args.emb_name}_nnls_mixture.csv")
        log(f"  wrote mixture {mix.shape}")

        reached = {}
        for t in labels:
            sel = mix[t] >= args.min_weight
            if sel.any():
                reached[t] = (int(sel.sum()), float(mix.loc[sel, t].max()))
        log(f"  labels with weight >= {args.min_weight} in some cluster: {len(reached)}")
        for t in labels:
            n, mx = reached.get(t, (0, 0.0))
            recovery_rows.append({
                "label": t, "n_clusters_supporting": n,
                "max_weight": round(mx, 4),
                "is_argmax_somewhere": bool((mix.values.argmax(1) == list(labels).index(t)).any()),
                "reached": n > 0,
            })

        # per-cell labels = the cluster's top mixture component, so downstream
        # (profile_niches) can consume this exactly like a baseline annotation
        top = pd.Series(labels[mix.values.argmax(1)], index=mix.index)
        per_cell = pd.DataFrame({"leiden": leiden.values,
                                 "annotation": leiden.map(top).values}, index=cells)
    else:
        # ---- per-cell transfer, optionally smoothed over the embedding's kNN
        if args.smooth_k > 1:
            log(f"Smoothing profiles over {args.smooth_k} embedding neighbours")
            nn = NearestNeighbors(n_neighbors=args.smooth_k).fit(emb)
            _, nbr = nn.kneighbors(emb)
            src_P = prof.values
            # Mean over the neighbourhood INCLUDING self (kneighbors returns self first).
            # Chunked: the naive src_P[nbr].mean(1) materialises cells x k x genes, which
            # is ~113 GB at k=50 on 193k cells x 2931 genes.
            P = np.empty_like(src_P)
            step = max(1, int(2e8 // (args.smooth_k * src_P.shape[1])))
            for s in range(0, len(nbr), step):
                e = min(s + step, len(nbr))
                P[s:e] = src_P[nbr[s:e]].mean(axis=1)
            del src_P
        else:
            log("No smoothing (--smooth-k 1): this is the embedding-INDEPENDENT "
                "gene-space argmax; all arms will agree by construction")
            P = prof.values
        cols = list(prof.columns)
        del prof

        # Pearson of every cell against every centroid, via centred/normalised dot
        log(f"Correlating {P.shape[0]} cells against {len(labels)} centroids")
        Pc = P - P.mean(axis=1, keepdims=True)
        Pn = Pc / np.where(np.linalg.norm(Pc, axis=1, keepdims=True) == 0, np.nan,
                           np.linalg.norm(Pc, axis=1, keepdims=True))
        T = ref_cent[cols].values if list(ref_cent.columns) != cols else ref_cent.values
        Tc = T - T.mean(axis=1, keepdims=True)
        Tn = Tc / np.linalg.norm(Tc, axis=1, keepdims=True)
        corr = np.nan_to_num(Pn, nan=0.0) @ Tn.T                 # cells x types
        best = corr.argmax(1)
        ann = labels[best]
        per_cell = pd.DataFrame({"leiden": "", "annotation": ann}, index=cells)

        vc = pd.Series(ann).value_counts()
        for t in labels:
            n = int(vc.get(t, 0))
            recovery_rows.append({
                "label": t, "n_cells": n,
                "cell_frac": round(n / len(ann), 5),
                "reached": n > 0,
            })
        log(f"  labels reached: {int((vc > 0).sum())}/{len(labels)}")

    per_cell.to_parquet(outdir / f"{args.tag}_{args.emb_name}_{args.method}_annotation.parquet")
    rec = pd.DataFrame(recovery_rows)
    rec.to_csv(outdir / f"{args.tag}_{args.emb_name}_{args.method}_recovery.csv", index=False)
    n_reached = int(rec["reached"].sum())
    log(f"DONE [{args.method}/{args.emb_name}/seed{args.seed}]: {n_reached}/{len(labels)} "
        f"reference labels reached")
    for t in ("EN-IT-L4-V1", "EN-IT-DL-1", "EN-IT-DL-2"):
        r = rec[rec["label"] == t]
        if len(r):
            log(f"  target {t}: reached={bool(r['reached'].iloc[0])}")


if __name__ == "__main__":
    main()
