#!/usr/bin/env python
"""
Is scGPT's poor LOCAL neighbourhood structure a fixable property of the coordinate
system, or intrinsic to the embedding?

WHY THIS EXISTS
---------------
embedding_discriminability.py established a specific split: on the contested EN-IT
subtypes, scGPT's latent is almost perfectly separable GLOBALLY (cross-validated
ROC-AUC 0.977-0.9998 against a ~0.50 shuffle null) while its LOCAL structure is poor
(kNN purity median 0.660 vs scVI's 0.992; 0.16-0.60 on EN-IT labels). Leiden builds
communities from the kNN graph, so it inherits the local structure, not the global
separability -- which is why the transfer's clusters come out mixed and the diluted
labels are never assigned.

That diagnosis leaves one cheap question unanswered before any transfer is rewritten:
how much of the local disorder is just the COORDINATE SYSTEM? A high-dimensional
latent with correlated, wildly unequally-scaled dimensions can be globally separable
yet have Euclidean kNN dominated by a few high-variance nuisance directions. If so,
whitening or rescaling would lift purity for free, Leiden would produce purer
clusters, and the expensive transfer rewrites (NNLS deconvolution, per-cell smoothed
transfer) might be unnecessary. If purity does NOT move, the disorder is real
geometry and the transfer must change.

This script answers only that question. It does not rewrite the transfer.

WHAT IT MEASURES
----------------
Per (embedding, variant), over the SAME proxy labels used by the discriminability
probe (reused from its output parquet, never recomputed, so the label definition
cannot drift between the two scripts):

  knn_purity            mean fraction of a cell's k neighbours sharing its label
  purity_enrichment     purity / label prevalence (a rare label has a low floor)
  knn_majority_recall   fraction of a label's cells whose kNN MAJORITY is that label.
                        This is the quantity closest to what Leiden can do: a label
                        that cannot win a local majority vote will not survive
                        community detection, whatever its mean purity.

VARIANTS
--------
  raw            Euclidean on the latent as-is (the pipeline's current behaviour)
  zscore         per-dimension standardisation, then Euclidean
  whiten         PCA-whitening: decorrelate + unit-variance every component
  pca50          PCA to 50 components, no whitening (tests dimensionality alone --
                 scGPT's 512 dims vs scVI's 30 is itself a candidate explanation)
  pca50_whiten   both
  cosine         cosine distance on the raw latent (scale-free direction-only)
  zscore_cosine  standardise, then cosine

COMPARISONS ARE WITHIN-SUBSAMPLE, NOT AGAINST PUBLISHED NUMBERS
---------------------------------------------------------------
kNN purity depends on sampling density, so values here will not match the published
full-cell knn_purity.csv when --subsample is on. `raw` is therefore ALWAYS recomputed
in the same subsample as every other variant, and the only valid reading is
variant-vs-raw within one run. A variant "fixes" the geometry if it moves purity and
majority-recall on the EN-IT targets toward scVI's raw values -- not if it merely
differs from the published table.

Outputs:
  geometry_purity.csv     one row per (arm, variant, subset, label)
  geometry_summary.csv    one row per (arm, variant, subset): medians + target means
  geometry_targets.csv    the target labels only, pivoted variant x arm, for reading

Runs in the neuro env (numpy/pandas/scikit-learn).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_named_paths(items, what):
    """['scvi=/p/a.parquet', ...] -> {'scvi': Path('/p/a.parquet')}."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--{what} expects name=path, got {item!r}")
        name, path = item.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"--{what} {name}: no such file {p}")
        out[name] = p
    return out


def transform(Z, variant, n_pca, seed):
    """Return (coordinates, sklearn metric) for a geometry variant.

    Transforms are fitted on the supplied (already subsampled) matrix: the question
    is whether the coordinate system helps, and a PCA/scaler fitted on the same cells
    whose neighbourhoods are then scored is the faithful version of 'rescale the
    latent before building the graph' as a pipeline would do it.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if variant == "raw":
        return Z, "euclidean"
    if variant == "cosine":
        return Z, "cosine"
    if variant == "zscore":
        return StandardScaler().fit_transform(Z), "euclidean"
    if variant == "zscore_cosine":
        return StandardScaler().fit_transform(Z), "cosine"
    if variant == "whiten":
        n = min(Z.shape[0], Z.shape[1])
        return PCA(n_components=n, whiten=True, random_state=seed).fit_transform(Z), "euclidean"
    if variant == "pca50":
        n = min(n_pca, Z.shape[0], Z.shape[1])
        return PCA(n_components=n, whiten=False, random_state=seed).fit_transform(Z), "euclidean"
    if variant == "pca50_whiten":
        n = min(n_pca, Z.shape[0], Z.shape[1])
        return PCA(n_components=n, whiten=True, random_state=seed).fit_transform(Z), "euclidean"
    raise SystemExit(f"unknown variant {variant!r}")


def main():
    p = argparse.ArgumentParser(
        description="Does rescaling/whitening the latent fix its local kNN structure?")
    p.add_argument("--proxy-parquet",
                   default="/scratch/cole.sy/neuro/results/embedding_discriminability/"
                           "per_cell_gene_space_label.parquet",
                   help="per-cell gene-space proxy labels from embedding_discriminability.py "
                        "(reused so the label definition is identical across the two scripts)")
    p.add_argument("--embedding", action="append", default=[], metavar="NAME=PATH",
                   help="per-cell embedding parquet, repeatable")
    p.add_argument("--variants",
                   default="raw,zscore,whiten,pca50,pca50_whiten,cosine,zscore_cosine")
    p.add_argument("--targets", default="EN-IT-L4-V1,EN-IT-DL-1,EN-IT-DL-2",
                   help="labels whose recovery is in question")
    p.add_argument("--knn", type=int, default=15, help="neighbors (matches pipeline default)")
    p.add_argument("--n-pca", type=int, default=50, help="components for the pca50* variants")
    p.add_argument("--subsample", type=int, default=60000,
                   help="cells to score (0 = all). Purity is density-dependent, so this "
                        "changes absolute values; every variant uses the SAME subsample, "
                        "so variant-vs-raw comparisons stay valid")
    p.add_argument("--min-cells", type=int, default=0,
                   help="skip labels with fewer than this many cells in the subsample "
                        "(0 = use --knn as the floor, matching the discriminability probe)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    from sklearn.neighbors import NearestNeighbors

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    embeddings = parse_named_paths(args.embedding, "embedding")
    if not embeddings:
        raise SystemExit("need at least one --embedding NAME=PATH")
    variants = [v for v in args.variants.split(",") if v]
    targets = [t for t in args.targets.split(",") if t]
    min_cells = args.min_cells or args.knn

    log(f"A. loading proxy labels {args.proxy_parquet}")
    proxy = pd.read_parquet(args.proxy_parquet)
    if "gene_space_label" not in proxy.columns:
        raise SystemExit(f"{args.proxy_parquet} has no 'gene_space_label' column")
    if "confident" not in proxy.columns:
        # The probe derives this within-label; recompute identically rather than
        # falling back to a global cutoff, which would empty the contested labels.
        grp = proxy.groupby("gene_space_label")["margin"]
        proxy["confident"] = proxy["margin"].values >= grp.transform("median").values
        log("   'confident' column absent; recomputed as margin >= per-label median")
    cells = proxy.index
    log(f"   {len(cells)} cells, {proxy['gene_space_label'].nunique()} proxy labels")

    log("B. loading embeddings")
    emb_arrays = {}
    for name, path in embeddings.items():
        edf = pd.read_parquet(path).reindex(cells)
        if edf.isna().any().any():
            raise SystemExit(f"{int(edf.isna().any(axis=1).sum())} cells missing from {path}")
        emb_arrays[name] = edf.to_numpy(dtype="float32")
        log(f"   {name}: {emb_arrays[name].shape}")

    # One subsample per subset, shared by EVERY arm and variant -- the comparison is
    # meaningless if the variants are scored on different cells.
    log(f"C. kNN structure per variant (k={args.knn}, subsample={args.subsample or 'all'})")
    rows = []
    for subset in ("all", "confident"):
        mask = np.ones(len(proxy), bool) if subset == "all" else proxy["confident"].values
        idx = np.flatnonzero(mask)
        if args.subsample and len(idx) > args.subsample:
            idx = np.sort(rng.choice(idx, args.subsample, replace=False))
        lab = proxy["gene_space_label"].values[idx]
        uniq = pd.unique(lab)
        log(f"   [{subset}] scoring {len(idx)} cells over {len(uniq)} labels")
        for name, emb in emb_arrays.items():
            Z0 = emb[idx]
            for variant in variants:
                t0 = time.time()
                Z, metric = transform(Z0, variant, args.n_pca, args.seed)
                nn = NearestNeighbors(n_neighbors=args.knn + 1, metric=metric,
                                      n_jobs=args.n_jobs).fit(Z)
                _, nbr = nn.kneighbors(Z)
                nbr = nbr[:, 1:]                                   # drop self
                nbr_lab = lab[nbr]
                same = (nbr_lab == lab[:, None]).mean(axis=1)
                # kNN majority label per cell: the local vote Leiden effectively inherits
                maj = pd.DataFrame(nbr_lab).mode(axis=1)[0].to_numpy()
                for label in uniq:
                    sel = lab == label
                    n = int(sel.sum())
                    if n < min_cells:
                        continue
                    prevalence = n / len(lab)
                    purity = float(same[sel].mean())
                    rows.append({
                        "arm": name, "variant": variant, "metric": metric,
                        "subset": subset, "label": label, "n_cells": n,
                        "prevalence": round(prevalence, 5),
                        "knn_purity": round(purity, 4),
                        "purity_enrichment": (round(purity / prevalence, 2)
                                              if prevalence > 0 else np.nan),
                        "knn_majority_recall": round(float((maj[sel] == label).mean()), 4),
                        "is_target": label in targets,
                    })
                log(f"     [{name}/{variant}/{subset}] {Z.shape[1]}d {metric} "
                    f"in {time.time() - t0:.0f}s")
    pur = pd.DataFrame(rows)
    pur.to_csv(outdir / "geometry_purity.csv", index=False)
    log(f"   wrote geometry_purity.csv ({len(pur)} rows)")

    # ---------- summary per (arm, variant, subset) ----------
    log("D. summarising")
    # Plain loop rather than groupby.apply: the apply form needs pandas>=2.2 for
    # include_groups, and this has to run in whichever env the arm's parquet came from.
    summ_rows = []
    for (arm, variant, metric, subset), g in pur.groupby(["arm", "variant", "metric", "subset"]):
        t = g[g["is_target"]]
        summ_rows.append({
            "arm": arm, "variant": variant, "metric": metric, "subset": subset,
            "n_labels": len(g),
            "median_purity": round(float(g["knn_purity"].median()), 4),
            "mean_purity": round(float(g["knn_purity"].mean()), 4),
            "median_majority_recall": round(float(g["knn_majority_recall"].median()), 4),
            "target_mean_purity": (round(float(t["knn_purity"].mean()), 4)
                                   if len(t) else np.nan),
            "target_mean_majority_recall": (round(float(t["knn_majority_recall"].mean()), 4)
                                            if len(t) else np.nan),
        })
    summ = pd.DataFrame(summ_rows)
    # Lift over this arm's OWN raw baseline -- the actionable number, since the
    # question is whether re-coordinating an embedding improves it, not whether one
    # embedding beats the other.
    base = (summ[summ["variant"] == "raw"]
            .set_index(["arm", "subset"])[["median_purity", "target_mean_purity",
                                           "target_mean_majority_recall"]])
    for col in ("median_purity", "target_mean_purity", "target_mean_majority_recall"):
        summ[f"{col}_vs_raw"] = (
            summ[col].values
            - base.reindex(pd.MultiIndex.from_arrays(
                [summ["arm"], summ["subset"]]))[col].values).round(4)
    summ.to_csv(outdir / "geometry_summary.csv", index=False)
    log(f"   wrote geometry_summary.csv ({len(summ)} rows)")

    tgt = pur[pur["is_target"]].copy()
    if not tgt.empty:
        tgt.pivot_table(index=["subset", "label", "variant"], columns="arm",
                        values=["knn_purity", "knn_majority_recall"]).to_csv(
            outdir / "geometry_targets.csv")
        log("   wrote geometry_targets.csv")

    # ---------- console view ----------
    log("SUMMARY: median kNN purity by variant (confident subset)")
    sc = summ[summ["subset"] == "confident"]
    print(sc[["arm", "variant", "metric", "median_purity", "median_purity_vs_raw",
              "target_mean_purity", "target_mean_purity_vs_raw",
              "target_mean_majority_recall"]].to_string(index=False), flush=True)

    log("SUMMARY: per-target kNN purity (confident subset)")
    if not tgt.empty:
        print(tgt[tgt["subset"] == "confident"]
              .pivot_table(index=["label", "variant"], columns="arm", values="knn_purity")
              .to_string(), flush=True)

    # Verdict: did any variant materially lift scGPT's target-label locality?
    for arm in sorted(summ["arm"].unique()):
        a = sc[sc["arm"] == arm]
        if a.empty:
            continue
        best = a.loc[a["target_mean_purity"].idxmax()]
        log(f"VERDICT [{arm}]: best variant for the target labels = {best['variant']} "
            f"(target purity {best['target_mean_purity']:.3f}, "
            f"{best['target_mean_purity_vs_raw']:+.3f} vs raw; majority-recall "
            f"{best['target_mean_majority_recall']:.3f})")
    log("Reading it: a large positive *_vs_raw for scGPT means the local disorder was a "
        "coordinate-system artifact and re-scaling before the kNN graph is the cheap fix "
        "-- re-run Leiden under that variant before rewriting the transfer. Lifts near "
        "zero mean the geometry is real and the transfer itself has to change.")
    log("DONE.")


if __name__ == "__main__":
    main()
