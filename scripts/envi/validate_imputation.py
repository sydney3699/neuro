#!/usr/bin/env python
"""
Quick sanity validation of the ENVI imputation outputs (no retraining).

Runs on the slim spatial AnnData written by run_envi.py
(FB080_spatial_envi.h5ad), which carries:
    X                       measured MERFISH counts (panel genes)
    obsm['imputation']      ENVI-imputed expression (cells x imputed genes)
    obsm['spatial']         spatial coordinates
    obsm['envi_latent']     spatial joint latent
    uns['imputation_genes'] column labels for obsm['imputation']
    obs                     H1/H2/H3_annotation, area, layer, ...

Produces (into <results>/validation/):
    1. insample_overlap_correlation.png + per_gene_correlation.csv
       Per-gene imputed-vs-measured correlation for genes measured in the
       panel AND imputed (in-sample upper bound; the rigorous leave-genes-out
       version lives in the k-fold harness).
    2. marker_celltype_heatmap.png   canonical markers x H1 class, mean imputed.
    3. spatial_markers.png           imputed layer markers on spatial coords.
    4. latent_umap_modality.png      joint spatial+sc latent, colored by modality.
    5. constant_genes.csv            imputed genes with collapsed dynamic range.

This is the fast, current-outputs pass (sanity #2/#3). It does not prove
imputation accuracy on held-out genes — see the leave-genes-out harness for that.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Canonical markers for developing human cortex -> expected H1 class.
# Only those present in the imputed gene set are used.
MARKERS = {
    "SLC17A7": "EN-IT", "SATB2": "EN-IT", "NEUROD2": "EN-IT", "NEUROD6": "EN-IT",
    "CUX2": "EN-IT", "RORB": "EN-IT", "BCL11B": "EN-ET", "FEZF2": "EN-ET", "TLE4": "EN-ET",
    "GAD1": "IN", "GAD2": "IN", "DLX2": "IN", "DLX5": "IN",
    "AQP4": "Astro", "GFAP": "Astro", "SLC1A3": "Astro",
    "AIF1": "MG", "CSF1R": "MG", "P2RY12": "MG",
    "CLDN5": "EC", "PECAM1": "EC",
    "VIM": "RG", "HES1": "RG", "PAX6": "RG", "SOX2": "RG",
    "EOMES": "IPC", "PPP1R17": "IPC",
    "OLIG1": "OPC", "OLIG2": "OPC", "PDGFRA": "OPC",
}
# Layer/area markers for the spatial coherence panel.
SPATIAL_MARKERS = ["RORB", "CUX2", "BCL11B", "FEZF2", "SATB2", "GAD1"]


def main():
    p = argparse.ArgumentParser(description="ENVI imputation sanity validation")
    p.add_argument("--h5ad", default="/scratch/cole.sy/neuro/results/envi_FB080/FB080_spatial_envi.h5ad")
    p.add_argument("--sc-latent", default="/scratch/cole.sy/neuro/results/envi_FB080/sc_envi_latent.npy",
                   help="sc joint latent (.npy) for the modality-mixing UMAP; skipped if absent.")
    p.add_argument("--outdir", default="/scratch/cole.sy/neuro/results/envi_FB080/validation")
    p.add_argument("--group-key", default="H1_annotation")
    p.add_argument("--area-key", default="area")
    p.add_argument("--n-sample", type=int, default=25000,
                   help="cells subsampled for correlation/UMAP (full data used for group means).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    import anndata as ad
    log(f"Loading {args.h5ad}")
    a = ad.read_h5ad(args.h5ad)
    log(f"  shape={a.shape}, obsm={list(a.obsm)}")

    imp_genes = np.asarray(a.uns["imputation_genes"], dtype=object)
    imp = a.obsm["imputation"]  # ndarray cells x imputed genes
    imp_idx = {g: i for i, g in enumerate(imp_genes)}
    log(f"  imputed genes={len(imp_genes)}, measured panel genes={a.n_vars}")

    # measured matrix as dense ndarray
    X = a.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    measured = list(a.var_names)
    n = a.n_obs
    samp = rng.choice(n, size=min(args.n_sample, n), replace=False)

    # ---- 1. In-sample overlap correlation (imputed vs measured) ----
    from scipy.stats import spearmanr
    shared = [g for g in measured if g in imp_idx]
    log(f"genes measured AND imputed: {len(shared)}/{len(measured)}")
    rows = []
    for g in shared:
        m = X[samp, measured.index(g)]
        q = imp[samp, imp_idx[g]]
        if m.std() == 0 or q.std() == 0:
            rho = np.nan
        else:
            rho = spearmanr(m, q).correlation
        rows.append({"gene": g, "spearman": rho})
    cor = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    cor.to_csv(outdir / "per_gene_correlation.csv", index=False)
    med = cor["spearman"].median()
    log(f"  median in-sample Spearman = {med:.3f} (n={cor['spearman'].notna().sum()} genes)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cor["spearman"].dropna(), bins=30, color="#4477AA", edgecolor="white")
    ax.axvline(med, color="crimson", lw=2, label=f"median={med:.3f}")
    ax.set(xlabel="per-gene Spearman (imputed vs measured)", ylabel="genes",
           title=f"In-sample overlap correlation (n={len(shared)} genes)")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "insample_overlap_correlation.png", dpi=150); plt.close(fig)

    # ---- 2. Marker x cell-type consistency ----
    if args.group_key in a.obs:
        groups = a.obs[args.group_key].astype(str).values
        markers = [g for g in MARKERS if g in imp_idx]
        log(f"markers present in imputed set: {len(markers)}/{len(MARKERS)}")
        uniq = sorted(pd.unique(groups))
        M = np.zeros((len(markers), len(uniq)))
        for j, grp in enumerate(uniq):
            mask = groups == grp
            M[:, j] = imp[mask][:, [imp_idx[g] for g in markers]].mean(axis=0)
        # z-score each marker (row) across cell types so enrichment is visible
        Mz = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-9)
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(uniq)), 0.32 * len(markers) + 2))
        im = ax.imshow(Mz, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(uniq))); ax.set_xticklabels(uniq, rotation=90, fontsize=7)
        ax.set_yticks(range(len(markers)))
        ax.set_yticklabels([f"{g} ({MARKERS[g]})" for g in markers], fontsize=7)
        ax.set_title(f"Imputed marker expression by {args.group_key} (row z-score)")
        fig.colorbar(im, ax=ax, shrink=0.5, label="z")
        fig.tight_layout(); fig.savefig(outdir / "marker_celltype_heatmap.png", dpi=150); plt.close(fig)
    else:
        log(f"  group key '{args.group_key}' absent; skipping marker heatmap")

    # ---- 3. Spatial coherence of imputed layer markers ----
    if "spatial" in a.obsm:
        xy = np.asarray(a.obsm["spatial"])[samp]
        panel = [g for g in SPATIAL_MARKERS if g in imp_idx]
        ncol = 3; nrow = int(np.ceil((len(panel) + 1) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
        axs = np.atleast_1d(axs).ravel()
        for k, g in enumerate(panel):
            v = imp[samp, imp_idx[g]]
            vmax = np.quantile(v, 0.99)
            sc = axs[k].scatter(xy[:, 0], xy[:, 1], c=v, s=2, cmap="viridis", vmax=vmax)
            axs[k].set_title(f"imputed {g}", fontsize=9); axs[k].set_aspect("equal"); axs[k].axis("off")
            fig.colorbar(sc, ax=axs[k], shrink=0.6)
        # reference panel: area labels
        if args.area_key in a.obs:
            areas = a.obs[args.area_key].astype(str).values[samp]
            for code in sorted(pd.unique(areas)):
                m = areas == code
                axs[len(panel)].scatter(xy[m, 0], xy[m, 1], s=2, label=code)
            axs[len(panel)].set_title("area (reference)", fontsize=9)
            axs[len(panel)].set_aspect("equal"); axs[len(panel)].axis("off")
            axs[len(panel)].legend(markerscale=4, fontsize=7, loc="best")
        for k in range(len(panel) + 1, len(axs)):
            axs[k].axis("off")
        fig.tight_layout(); fig.savefig(outdir / "spatial_markers.png", dpi=150); plt.close(fig)

    # ---- 4. Joint latent UMAP (modality mixing) ----
    sc_latent_path = Path(args.sc_latent)
    if "envi_latent" in a.obsm and sc_latent_path.exists():
        try:
            import scanpy as sc
            sp_lat = np.asarray(a.obsm["envi_latent"])[samp]
            sc_lat = np.load(sc_latent_path)
            sc_samp = rng.choice(sc_lat.shape[0], size=min(args.n_sample, sc_lat.shape[0]), replace=False)
            sc_lat = sc_lat[sc_samp]
            lat = np.vstack([sp_lat, sc_lat]).astype(np.float32)
            modality = np.array(["spatial"] * len(sp_lat) + ["snRNA"] * len(sc_lat))
            la = ad.AnnData(X=lat)
            la.obs["modality"] = modality
            sc.pp.neighbors(la, use_rep="X", n_neighbors=15)
            sc.tl.umap(la)
            u = la.obsm["X_umap"]
            fig, ax = plt.subplots(figsize=(6, 5))
            for mod, col in [("snRNA", "#EE6677"), ("spatial", "#4477AA")]:
                m = modality == mod
                ax.scatter(u[m, 0], u[m, 1], s=2, alpha=0.4, c=col, label=mod)
            ax.set(title="Joint ENVI latent (modality mixing)", xticks=[], yticks=[])
            ax.legend(markerscale=4)
            fig.tight_layout(); fig.savefig(outdir / "latent_umap_modality.png", dpi=150); plt.close(fig)
            log("  UMAP done")
        except Exception as e:
            log(f"  UMAP skipped ({e.__class__.__name__}: {e})")
    else:
        log("  sc latent or envi_latent missing; skipping UMAP")

    # ---- 5. Collapsed / near-constant imputed genes ----
    sub = imp[samp]
    g_std = sub.std(axis=0); g_mean = sub.mean(axis=0)
    cv = g_std / (np.abs(g_mean) + 1e-9)
    const = pd.DataFrame({"gene": imp_genes, "mean": g_mean, "std": g_std, "cv": cv})
    const = const.sort_values("cv")
    n_flag = int((cv < 0.05).sum())
    const.to_csv(outdir / "constant_genes.csv", index=False)
    log(f"  near-constant imputed genes (cv<0.05): {n_flag}/{len(imp_genes)}")

    log(f"DONE. Validation outputs in {outdir}")


if __name__ == "__main__":
    main()
