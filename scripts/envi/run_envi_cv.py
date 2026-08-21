#!/usr/bin/env python
"""
Leave-genes-out cross-validation of ENVI imputation (one fold per invocation).

Honest accuracy check: withhold a fold of the shared panel genes from the
SPATIAL side only (snRNA keeps everything), retrain ENVI, impute, and compare
the imputed values for the held-out genes against their true measured MERFISH
values. Because the held-out genes are absent from the spatial<->sc overlap,
ENVI never sees them during training for these cells -> genuine holdout.

Per held-out gene, Spearman(imputed, measured) is compared against two baselines
on the same eval cells:
  - knn        : neighbors in measured panel-gene space; predict gene = mean of
                 neighbors' measured held-out values (strong baseline).
  - celltype   : predict gene = its per-cell-type mean (trivial floor).

Writes one CSV per fold: cv_fold<k>.csv with columns
  fold, gene, n_eval, envi, knn, celltype, imputed_present
An array job (envi_cv.sbatch) runs folds 0..n-1; aggregate_cv.py combines them.
"""
import argparse
import os
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="ENVI leave-genes-out CV (single fold)")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--spatial", default="/scratch/cole.sy/neuro/data/processed/FB080_O1b_v1v2_envi_input.h5ad")
    p.add_argument("--snrna", default="/scratch/cole.sy/neuro/data/raw/snrna.h5ad")
    p.add_argument("--outdir", default="/scratch/cole.sy/neuro/results/envi_cv")
    p.add_argument("--training-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-hvg", type=int, default=2048)
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--celltype-key", default="H2_annotation")
    p.add_argument("--n-eval", type=int, default=30000, help="cells subsampled for the gene-wise eval.")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--split-seed", type=int, default=0, help="seed for the gene fold split (SAME across folds).")
    p.add_argument("--seed", type=int, default=0, help="ENVI training key seed.")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import jax
    devs = jax.devices()
    log(f"JAX devices: {devs}")
    if not any(d.platform == "gpu" for d in devs) and not args.allow_cpu:
        raise SystemExit("No GPU visible to JAX. Run on a GPU node or pass --allow-cpu.")

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from jax import random
    from scenvi import ENVI
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    # --- Load + reconcile aliases (mirror run_envi.py) ---
    spatial = ad.read_h5ad(args.spatial)
    sc = ad.read_h5ad(args.snrna)
    sc.var_names_make_unique(); sc.obs_names_make_unique()
    alias_map = {"DENND2B": "ST5", "ELAPOR1": "KIAA1324", "FLJ20021": "TMEM131L"}
    sc_genes = set(sc.var_names); panel = set(spatial.var_names)
    spatial.var_names = [alias_map[g] if (g in alias_map and alias_map[g] in sc_genes
                                          and alias_map[g] not in panel) else g
                         for g in spatial.var_names]

    # --- Deterministic gene fold split over the shared genes (same for every fold) ---
    shared = sorted(set(spatial.var_names) & sc_genes)
    order = np.random.default_rng(args.split_seed).permutation(len(shared))
    folds = np.array_split(order, args.n_folds)
    held_idx = folds[args.fold]
    held = [shared[i] for i in held_idx]
    log(f"fold {args.fold}/{args.n_folds}: {len(shared)} shared genes, holding out {len(held)}")
    log(f"  held-out: {held}")

    # --- Truth for held-out genes (before we drop them from spatial) ---
    def dense(a):
        return np.asarray(a.todense(), dtype=np.float32) if sp.issparse(a) else np.asarray(a, np.float32)
    truth = dense(spatial[:, held].X)                      # cells x held
    groups = spatial.obs[args.celltype_key].astype(str).values

    # --- Drop held-out genes from spatial; keep panel-train matrix for kNN ---
    train_genes = [g for g in spatial.var_names if g not in set(held)]
    spatial_train = spatial[:, train_genes].copy()
    panel_train = dense(spatial_train.X)                   # cells x train-panel

    # densify for scenvi
    for a in (spatial_train, sc):
        if sp.issparse(a.X):
            a.X = np.asarray(a.X.todense(), dtype=np.float32)

    # --- Train ENVI on the reduced overlap ---
    log("Init + train ENVI ...")
    t0 = time.time()
    model = ENVI(spatial_data=spatial_train, sc_data=sc,
                 spatial_key=args.spatial_key, num_HVG=args.num_hvg)
    model.train(training_steps=args.training_steps, batch_size=args.batch_size,
                key=random.key(args.seed))
    model.latent_rep()
    model.impute_genes()
    log(f"  trained + imputed in {time.time()-t0:.1f}s")
    imp = model.spatial_data.obsm["imputation"]            # DataFrame cells x sc HVG
    imp_cols = {g: j for j, g in enumerate(imp.columns)}
    imp_vals = imp.to_numpy(dtype=np.float32)

    # --- Eval subsample (shared across all three predictors) ---
    n = spatial.n_obs
    rng = np.random.default_rng(args.seed)
    ev = rng.choice(n, size=min(args.n_eval, n), replace=False)

    # kNN in log1p panel-train space; neighbor indices computed once, reused per gene
    Xtr = np.log1p(panel_train[ev])
    nn = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(Xtr)
    _, nbr = nn.kneighbors(Xtr)
    nbr = nbr[:, 1:]                                        # drop self

    rows = []
    for gi, g in enumerate(held):
        y = truth[ev, gi]
        if y.std() == 0:
            rows.append(dict(fold=args.fold, gene=g, n_eval=len(ev),
                             envi=np.nan, knn=np.nan, celltype=np.nan, imputed_present=g in imp_cols))
            continue
        # ENVI
        envi_rho = np.nan
        if g in imp_cols:
            q = imp_vals[ev, imp_cols[g]]
            envi_rho = spearmanr(q, y).correlation if q.std() > 0 else np.nan
        # kNN: mean of neighbors' measured held-out values
        knn_pred = y[nbr].mean(axis=1)
        knn_rho = spearmanr(knn_pred, y).correlation if knn_pred.std() > 0 else np.nan
        # cell-type mean (floor)
        ge = groups[ev]
        ser = pd.Series(y).groupby(ge).transform("mean").to_numpy()
        ct_rho = spearmanr(ser, y).correlation if np.std(ser) > 0 else np.nan
        rows.append(dict(fold=args.fold, gene=g, n_eval=len(ev),
                         envi=envi_rho, knn=knn_rho, celltype=ct_rho, imputed_present=g in imp_cols))

    out = Path(args.outdir) / f"cv_fold{args.fold}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"DONE fold {args.fold}: wrote {out}")


if __name__ == "__main__":
    main()
