#!/usr/bin/env python
"""
ENVI integration of the FB080 V1/V2 MERFISH spatial data with the paired
snRNA-seq reference, imputing transcriptome-wide expression onto the spatial cells.

Uses the standalone JAX/Flax `scenvi` package (GPU via the cuda jaxlib build).
Intended to run on a GPU node — see scripts/envi_gpu.sbatch.

Workflow (scenvi 0.4.6):
    model = ENVI(spatial_data, sc_data, spatial_key="spatial")  # computes COVET on init
    model.train(training_steps=...)                             # GPU training
    model.latent_rep()                                          # obsm['envi_latent'] (required pre-impute)
    model.impute_genes()                                        # spatial.obsm['imputation']
    model.infer_niche_covet(); model.infer_niche_celltype(...)  # optional niche analysis
"""
import argparse
import os
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="ENVI imputation: FB080 MERFISH x snRNA-seq")
    p.add_argument("--spatial", default="/scratch/cole.sy/neuro/data/processed/FB080_O1b_v1v2_envi_input.h5ad")
    p.add_argument("--snrna", default="/scratch/cole.sy/neuro/data/raw/snrna.h5ad")
    p.add_argument("--outdir", default="/scratch/cole.sy/neuro/results/envi_FB080")
    p.add_argument("--training-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-hvg", type=int, default=2048,
                   help="HVGs kept for the snRNA reference; imputation is onto this gene set + spatial overlap.")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--celltype-key", default=None,
                   help="spatial obs column for niche cell-type inference (e.g. H3_annotation). "
                        "If set, runs infer_niche_covet + infer_niche_celltype.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true",
                   help="Proceed even if no GPU is visible to JAX (training will be very slow).")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- GPU check first: fail fast rather than burn hours on CPU ---
    import jax
    devs = jax.devices()
    log(f"JAX devices: {devs}")
    has_gpu = any(d.platform == "gpu" for d in devs)
    if not has_gpu and not args.allow_cpu:
        raise SystemExit("No GPU visible to JAX. Run on a GPU node, or pass --allow-cpu to override.")

    import anndata as ad
    import numpy as np
    from jax import random
    from scenvi import ENVI

    # --- Load data ---
    log(f"Loading spatial: {args.spatial}")
    spatial = ad.read_h5ad(args.spatial)
    log(f"  spatial shape={spatial.shape}, obsm={list(spatial.obsm)}")
    if args.spatial_key not in spatial.obsm:
        raise SystemExit(f"spatial_key '{args.spatial_key}' not in spatial.obsm ({list(spatial.obsm)})")

    log(f"Loading snRNA: {args.snrna}")
    sc = ad.read_h5ad(args.snrna)
    sc.var_names_make_unique()
    sc.obs_names_make_unique()
    log(f"  snRNA shape={sc.shape}")

    # --- Reconcile HGNC alias drift between MERFISH panel and snRNA reference ---
    # (mirrors notebooks/04_envi_imputation). Idempotent: only renames an old symbol
    # if its current symbol is present in the snRNA reference and not already in the panel.
    alias_map = {"DENND2B": "ST5", "ELAPOR1": "KIAA1324", "FLJ20021": "TMEM131L"}
    sc_genes = set(sc.var_names)
    panel = set(spatial.var_names)
    renamed, new_names = [], []
    for g in spatial.var_names:
        ng = alias_map.get(g, g)
        if ng != g and ng in sc_genes and ng not in panel:
            renamed.append((g, ng))
            new_names.append(ng)
        else:
            new_names.append(g)
    spatial.var_names = new_names
    log(f"  alias renames applied: {renamed}")
    overlap = set(spatial.var_names) & sc_genes
    log(f"  spatial∩snRNA overlap: {len(overlap)}/{spatial.n_vars}")

    # --- Sanity: scenvi expects raw counts for pois/nb distributions ---
    for name, a in [("spatial", spatial), ("snRNA", sc)]:
        x = a.X[:200]
        x = x.toarray() if hasattr(x, "toarray") else x
        log(f"  {name} .X integer counts: {bool((x % 1 == 0).all())} "
            f"(min={float(a.X.min())}, max={float(a.X.max())})")

    # scenvi assumes dense .X: it does `np.log(X + 1)` (scipy sparse rejects scalar add)
    # and indexes .X directly in the training loop. Densify to float32 up front.
    import scipy.sparse as sp
    for name, a in [("spatial", spatial), ("snRNA", sc)]:
        if sp.issparse(a.X):
            log(f"  densifying {name} .X {a.shape} -> float32 ndarray")
            a.X = np.asarray(a.X.todense(), dtype=np.float32)

    # --- ENVI ---
    log("Initializing ENVI (computes COVET niche matrices) ...")
    t0 = time.time()
    model = ENVI(spatial_data=spatial, sc_data=sc,
                 spatial_key=args.spatial_key, num_HVG=args.num_hvg)
    log(f"  ENVI init done in {time.time()-t0:.1f}s; overlap_num={model.overlap_num}, "
        f"sc genes kept={model.sc_data.n_vars}")

    log(f"Training {args.training_steps} steps (batch_size={args.batch_size}) ...")
    t0 = time.time()
    model.train(training_steps=args.training_steps, batch_size=args.batch_size,
                key=random.key(args.seed))
    log(f"  training done in {time.time()-t0:.1f}s")

    log("Computing latent representations (latent_rep) ...")
    model.latent_rep()
    log("Imputing transcriptome onto spatial cells (impute_genes) ...")
    model.impute_genes()

    niche_ran = False
    if args.celltype_key:
        if args.celltype_key not in model.spatial_data.obs.columns:
            raise SystemExit(f"--celltype-key '{args.celltype_key}' not in spatial obs "
                             f"({list(model.spatial_data.obs.columns)})")
        # Wrapped so a niche failure can't discard the imputation outputs below.
        try:
            log(f"Inferring niche cell-type composition (key={args.celltype_key}) ...")
            model.infer_niche_covet()
            # scenvi's niche_cell_type() hardcodes obs['cell_type'] in one line, so
            # mirror the chosen annotation column there before calling.
            model.spatial_data.obs["cell_type"] = model.spatial_data.obs[args.celltype_key].values
            model.infer_niche_celltype(cell_type_key="cell_type")
            niche_ran = True
            log("  niche inference done")
        except Exception as e:
            log(f"  niche inference FAILED ({e.__class__.__name__}: {e}) — continuing to save imputation")

    # --- Save outputs (each guarded independently so one failure can't lose the rest) ---
    outdir = Path(args.outdir)
    imp = model.spatial_data.obsm["imputation"]  # DataFrame: spatial cells x imputed genes
    log(f"imputation shape={imp.shape}")

    def save(label, fn):
        try:
            fn()
            log(f"  saved {label}")
        except Exception as e:
            log(f"  FAILED to save {label}: {e.__class__.__name__}: {e}")

    # Imputation: prefer parquet, fall back to compressed npz (no pyarrow needed).
    def _imp():
        try:
            imp.to_parquet(outdir / "FB080_envi_imputation.parquet")
        except Exception as e:
            log(f"  parquet engine unavailable ({e.__class__.__name__}); writing .npz instead")
            np.savez_compressed(
                outdir / "FB080_envi_imputation.npz",
                values=imp.to_numpy(dtype=np.float32),
                genes=np.asarray(imp.columns, dtype=object),
                cells=np.asarray(imp.index, dtype=object),
            )
    save("imputation", _imp)
    save("spatial latent", lambda: np.save(outdir / "spatial_envi_latent.npy",
                                           np.asarray(model.spatial_data.obsm["envi_latent"])))
    save("sc latent", lambda: np.save(outdir / "sc_envi_latent.npy",
                                      np.asarray(model.sc_data.obsm["envi_latent"])))
    save("spatial obs", lambda: model.spatial_data.obs.to_csv(outdir / "spatial_obs.csv"))

    if niche_ran:
        save("spatial cell_type_niche",
             lambda: model.spatial_data.obsm["cell_type_niche"].to_csv(outdir / "spatial_cell_type_niche.csv"))
        save("sc cell_type_niche",
             lambda: model.sc_data.obsm["cell_type_niche"].to_csv(outdir / "sc_cell_type_niche.csv"))

    # Slim spatial AnnData carrying obsm['imputation'] + envi_latent; drop bulky 3D COVET arrays.
    def _h5ad():
        sp_out = model.spatial_data.copy()
        for k in ["COVET", "COVET_SQRT"]:
            sp_out.obsm.pop(k, None)
        sp_out.write_h5ad(outdir / "FB080_spatial_envi.h5ad")
    save("spatial h5ad", _h5ad)

    log(f"DONE. Outputs in {outdir}")


if __name__ == "__main__":
    main()
