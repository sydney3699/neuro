#!/usr/bin/env python
"""
Per-cell probe: does an embedding ENCODE a cell-type distinction that the
Leiden + centroid label transfer never assigns?

WHY THIS EXISTS
---------------
label_recovery.py establishes a transfer-level fact: the scGPT arm assigns no
cluster to EN-IT-L4-V1 at any Leiden resolution from 1.0 to 4.0 -- including at 36
and 42 clusters, where its granularity matches or exceeds the scVI arm's 37 -- while
the scVI arm reciprocally never reaches EN-IT-DL-2. That rules out cluster count as
the explanation but CANNOT distinguish three remaining causes:

  (a) the embedding does not separate those cells at all   <- an embedding claim
  (b) it separates them, but Leiden merges them into mixed clusters
  (c) it separates them and Leiden splits them, but every cluster centroid's argmax
      Pearson still lands on a neighbouring reference type

This script separates (a) from (b)/(c) by measuring the embedding DIRECTLY, with no
clustering step in the loop.

THE GROUND-TRUTH PROBLEM, AND WHAT WE DO ABOUT IT
-------------------------------------------------
The spatial cells have no per-cell ground-truth labels -- producing them is what the
annotation pipeline is for, so using its output as truth would be circular. And the
reference cells cannot simply be embedded alongside: scVI here is trained on the
ENVI-imputed SPATIAL cells only, and the snRNA reference enters solely as gene-space
centroids, so there is no scVI latent coordinate for a reference cell without model
surgery or retraining (a different model than the one that produced the results).

So the target is a PROXY: per-cell gene-space reference affinity. Each spatial cell's
log1p-CP10k profile over the shared genes is correlated against every reference
centroid, and its argmax is taken as that cell's provisional type. This proxy is
computed once, identically for both arms, from expression only -- it is
EMBEDDING-INDEPENDENT, which is the single property the comparison requires. It is
NOT biological truth: the correlation runs over the ~2.9k shared genes, but those are
ENVI imputations driven by a 300-gene measured panel, so per-cell profiles carry far
less independent information than the gene count suggests, and transitional states in
developing cortex genuinely sit between centroids. Hence
every metric is also reported on the confident half (top 50% best-minus-second
margin), and `--validate-transfer` checks that this script's normalization
reproduces the published cluster-level labels exactly before any of it is believed.

WHAT EACH STAGE SHOWS
---------------------
  A  per-cell gene-space proxy labels (+ margin), embedding-independent
  A2 validation: recompute the cluster-level argmax from the published leiden
     assignments and confirm it reproduces the published `annotation` column
  B  kNN purity: do cells sharing a proxy label sit together in this embedding?
     Reported against label prevalence as an enrichment, since a rare label has a
     low purity floor. Diffuse (enrichment ~1) => cause (a). Concentrated => (b)/(c).
  C  pairwise discriminability: cross-validated ROC-AUC / macro-F1 for the contested
     pair (e.g. EN-IT-L4-V1 vs EN-IT-UL-1) on balanced subsets, against a
     label-shuffle null. AUC near the null => the embedding carries no signal
     separating them; AUC well above it => the information is present and the loss
     happened downstream.
  D  cluster dilution: where the proxy-labelled cells actually land among the
     published Leiden clusters -- concentration and the maximum within-cluster
     fraction. A label whose cells never exceed a small fraction of any cluster
     explains a never-assigned centroid argmax directly, and distinguishes (b)/(c).

INTERPRETATION LIMIT, STATED ONCE
---------------------------------
High AUC + low purity + low within-cluster fraction = the distinction is in the
latent space but the transfer pipeline discards it -> fix the transfer, do not
conclude the embedding is deficient. Low AUC on the confident subset, with the
companion arm scoring high on the same pair, is the only result here that supports
an embedding-level claim -- and even then, scVI was trained on these cells while
scGPT is zero-shot, so any such gap is partly a trained-vs-zero-shot gap, not a
clean architecture comparison.

Runs in the neuro env (anndata/scanpy-free: numpy/pandas/scikit-learn + anndata).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from neurospatial.expression import corr_to_centroids, lognorm_profiles


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


def main():
    p = argparse.ArgumentParser(
        description="Per-cell embedding discriminability probe for contested cell-type labels")
    p.add_argument("--spatial-h5ad",
                   default="/scratch/cole.sy/neuro/results/envi_FB080_lr/FB080_spatial_envi.h5ad",
                   help="ENVI output; expression read from obsm['imputation'] + uns['imputation_genes']")
    p.add_argument("--reference-h5ad", default="/scratch/cole.sy/neuro/data/raw/snrna.h5ad")
    p.add_argument("--ref-label-key", default="celltypes")
    p.add_argument("--ref-centroids-csv", default="",
                   help="precomputed reference centroids (index=label, columns=shared genes, "
                        "values=mean log1p-CP10k) instead of --reference-h5ad; build with "
                        "scripts/annotation/build_ref_centroids.py")
    p.add_argument("--embedding", action="append", default=[], metavar="NAME=PATH",
                   help="per-cell embedding parquet, repeatable "
                        "(e.g. scgpt=results/annot_scgpt/FB080_scgpt_embedding.parquet)")
    p.add_argument("--annotation", action="append", default=[], metavar="NAME=PATH",
                   help="per-cell annotation parquet (leiden + annotation), repeatable; "
                        "enables stages A2 and D for that arm")
    p.add_argument("--targets", default="EN-IT-L4-V1,EN-IT-DL-1,EN-IT-DL-2",
                   help="labels whose recovery is in question")
    p.add_argument("--pairs", default="EN-IT-L4-V1:EN-IT-UL-1,EN-IT-DL-1:EN-IT-DL-3,EN-IT-DL-2:EN-IT-DL-3",
                   help="comma-separated A:B pairs to test for pairwise discriminability")
    p.add_argument("--knn", type=int, default=15, help="neighbors for purity (matches pipeline default)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--max-per-class", type=int, default=4000,
                   help="cap per class in the balanced pairwise subsets (runtime control)")
    p.add_argument("--n-perm", type=int, default=50,
                   help="label-shuffle repeats for the null AUC (each is a full CV). Sets the "
                        "p-value floor at 1/(n_perm+1), so 10 perms cannot report below 0.091; "
                        "use >=99 if you need p<0.01")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="parallel workers for the permutation null (-1 = all cores)")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="LogisticRegression max_iter; dominates cost on high-dim embeddings")
    p.add_argument("--purity-subsample", type=int, default=0,
                   help="subsample cells for the kNN purity graph (0 = all cells)")
    p.add_argument("--validate-transfer", action="store_true",
                   help="stage A2: recompute cluster-level argmax and compare to the published labels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    embeddings = parse_named_paths(args.embedding, "embedding")
    annotations = parse_named_paths(args.annotation, "annotation")
    if not embeddings:
        raise SystemExit("need at least one --embedding NAME=PATH")
    targets = [t for t in args.targets.split(",") if t]
    pairs = [tuple(x.split(":")) for x in args.pairs.split(",") if ":" in x]

    # ---------- A. per-cell gene-space proxy labels (embedding-independent) ----------
    log(f"A. per-cell gene-space reference affinity")
    log(f"   loading {args.spatial_h5ad}")
    src = ad.read_h5ad(args.spatial_h5ad)
    genes = [str(g) for g in src.uns["imputation_genes"]]
    X = np.asarray(src.obsm["imputation"], dtype="float32")
    cells = pd.Index(src.obs_names)
    del src
    log(f"   {len(cells)} cells x {len(genes)} imputation genes")

    if args.ref_centroids_csv:
        # Precomputed centroids: index = reference label, columns = the shared genes,
        # values = mean log1p-CP10k (i.e. the output of lognorm_mean_by). Lets the probe
        # run where the multi-GB reference h5ad is not available, since the transfer only
        # ever consumes these ~N_types x ~300 means. The CSV's columns ARE the shared gene
        # set, so the same transform must have produced it -- build it with
        # neurospatial.expression.lognorm_mean_by, not by hand.
        log(f"   loading precomputed reference centroids {args.ref_centroids_csv}")
        ref_cent = pd.read_csv(args.ref_centroids_csv, index_col=0)
        shared = [str(c) for c in ref_cent.columns]
        unknown = [g for g in shared if g not in set(genes)]
        if unknown:
            raise SystemExit(f"{len(unknown)} centroid genes absent from the imputation "
                             f"genes, e.g. {unknown[:5]}")
        log(f"   {len(ref_cent)} centroids x {len(shared)} shared genes")
    else:
        log(f"   loading reference {args.reference_h5ad}")
        ref = ad.read_h5ad(args.reference_h5ad)
        if args.ref_label_key not in ref.obs:
            raise SystemExit(f"reference has no obs['{args.ref_label_key}']")
        ref_vars = [str(v) for v in ref.var_names]
        shared = [g for g in genes if g in set(ref_vars)]
        log(f"   {len(shared)} shared genes")
        ref_prof = lognorm_profiles(ref.X, ref_vars, shared)
        ref_prof["_g"] = ref.obs[args.ref_label_key].astype(str).values
        ref_cent = ref_prof.groupby("_g").mean()
        del ref, ref_prof
        log(f"   {len(ref_cent)} reference centroids")

    cell_prof = lognorm_profiles(X, genes, shared)
    corr = corr_to_centroids(cell_prof.values, ref_cent)
    del cell_prof
    order = np.argsort(-np.nan_to_num(corr, nan=-np.inf), axis=1)
    ridx = np.arange(corr.shape[0])
    best, second = order[:, 0], order[:, 1]
    proxy = pd.DataFrame({
        "gene_space_label": ref_cent.index.to_numpy()[best],
        "best_corr": corr[ridx, best],
        "second_label": ref_cent.index.to_numpy()[second],
        "margin": corr[ridx, best] - corr[ridx, second],
    }, index=cells)
    proxy.to_parquet(outdir / "per_cell_gene_space_label.parquet")
    vc = proxy["gene_space_label"].value_counts()
    log(f"   wrote per_cell_gene_space_label.parquet; {len(vc)} labels assigned")
    for t in targets:
        log(f"     target {t}: {int(vc.get(t, 0))} cells "
            f"({100.0 * vc.get(t, 0) / len(proxy):.2f}%)")
    # Confident half, thresholded WITHIN each proxy label -- not globally. A global
    # margin cutoff would discard almost every cell of exactly the labels under test:
    # closely-related subtypes have small achievable margins by construction, so the
    # global median is set by the easily-separated types (IPC, glia) and the contested
    # pair falls entirely below it, leaving the robustness check empty for the only
    # labels it needed to cover. This is the same failure mode as the fixed
    # `margin < 0.05` cluster flag that compute_low_confidence replaced with a
    # within-run percentile.
    grp = proxy.groupby("gene_space_label")["margin"]
    proxy["confident"] = proxy["margin"].values >= grp.transform("median").values
    log(f"   confident subset = margin >= per-label median ({int(proxy['confident'].sum())} cells)")
    for t in targets:
        sel = proxy["gene_space_label"].values == t
        if sel.any():
            log(f"     {t}: per-label median margin {float(np.nanmedian(proxy['margin'].values[sel])):.4f}, "
                f"{int((sel & proxy['confident'].values).sum())} confident cells")
    vc.rename_axis("label").reset_index(name="n_cells").to_csv(
        outdir / "gene_space_label_counts.csv", index=False)

    # ---------- A2. validate that this normalization reproduces published labels ----------
    if args.validate_transfer and annotations:
        log("A2. validating normalization against the published cluster-level transfer")
        rows = []
        for name, path in annotations.items():
            ann = pd.read_parquet(path).reindex(cells)
            if "leiden" not in ann.columns:
                log(f"   SKIP {name}: annotation parquet has no 'leiden' column")
                continue
            prof = lognorm_profiles(X, genes, shared)
            prof["_g"] = ann["leiden"].astype(str).values
            clus_mean = prof.groupby("_g").mean()
            del prof
            c = corr_to_centroids(clus_mean.values, ref_cent)
            recomputed = pd.Series(ref_cent.index.to_numpy()[c.argmax(1)], index=clus_mean.index)
            published = ann.groupby(ann["leiden"].astype(str))["annotation"].first()
            common = recomputed.index.intersection(published.index)
            agree = int((recomputed.loc[common].values == published.loc[common].values).sum())
            rows.append({"arm": name, "n_clusters": len(common), "n_agree": agree,
                         "frac_agree": round(agree / max(len(common), 1), 4)})
            log(f"   {name}: {agree}/{len(common)} clusters reproduce the published label "
                f"({agree / max(len(common), 1):.1%})")
            if agree < len(common):
                mism = [(i, recomputed[i], published[i]) for i in common
                        if recomputed[i] != published[i]]
                for i, r_, p_ in mism[:5]:
                    log(f"     cluster {i}: recomputed {r_} vs published {p_}")
        pd.DataFrame(rows).to_csv(outdir / "transfer_validation.csv", index=False)
        log("   NOTE: <100% agreement means this probe's normalization differs from the "
            "published run -- investigate before trusting stages B-D.")

    # ---------- B. kNN purity per label, per embedding ----------
    log("B. kNN purity of proxy labels within each embedding")
    emb_arrays = {}
    for name, path in embeddings.items():
        edf = pd.read_parquet(path).reindex(cells)
        if edf.isna().any().any():
            raise SystemExit(f"{int(edf.isna().any(axis=1).sum())} cells missing from {path}")
        emb_arrays[name] = edf.to_numpy(dtype="float32")
        log(f"   {name}: {emb_arrays[name].shape}")

    pur_rows = []
    for subset in ("all", "confident"):
        mask = np.ones(len(proxy), bool) if subset == "all" else proxy["confident"].values
        idx_all = np.flatnonzero(mask)
        if args.purity_subsample and len(idx_all) > args.purity_subsample:
            idx_all = np.sort(rng.choice(idx_all, args.purity_subsample, replace=False))
        lab = proxy["gene_space_label"].values[idx_all]
        for name, emb in emb_arrays.items():
            nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(emb[idx_all])
            _, nbr = nn.kneighbors(emb[idx_all])
            nbr = nbr[:, 1:]                                   # drop self
            same = (lab[nbr] == lab[:, None]).mean(axis=1)
            for label in pd.unique(lab):
                sel = lab == label
                n = int(sel.sum())
                if n < args.knn:
                    continue
                prevalence = n / len(lab)
                purity = float(same[sel].mean())
                pur_rows.append({
                    "arm": name, "subset": subset, "label": label, "n_cells": n,
                    "prevalence": round(prevalence, 5),
                    "knn_purity": round(purity, 4),
                    # purity relative to chance: 1.0 = indistinguishable from random
                    "purity_enrichment": round(purity / prevalence, 2) if prevalence > 0 else np.nan,
                    "is_target": label in targets,
                })
    pur = pd.DataFrame(pur_rows)
    pur.to_csv(outdir / "knn_purity.csv", index=False)
    log(f"   wrote knn_purity.csv ({len(pur)} rows)")
    tp = pur[(pur["is_target"]) & (pur["subset"] == "confident")]
    for r in tp.itertuples():
        log(f"     [{r.arm}] {r.label}: purity {r.knn_purity:.3f} "
            f"(prevalence {r.prevalence:.4f}, enrichment {r.purity_enrichment}x, n={r.n_cells})")

    # ---------- C. pairwise discriminability vs a label-shuffle null ----------
    log("C. pairwise discriminability (CV logistic regression vs label-shuffle null)")

    def cv_scores(Z, y, seed):
        """Stratified-CV out-of-fold ROC-AUC and macro-F1 for a binary problem."""
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
        proba, pred = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in skf.split(Z, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=args.max_iter,
                                                   class_weight="balanced"))
            clf.fit(Z[tr], y[tr])
            proba[te] = clf.predict_proba(Z[te])[:, 1]
            pred[te] = clf.predict(Z[te])
        return roc_auc_score(y, proba), f1_score(y, pred, average="macro")

    def null_aucs(Z, y):
        """Label-shuffle null AUCs, fitted in parallel.

        This dominates runtime: each permutation is a full CV, and scGPT's 512-dim
        latent makes a single lbfgs fit ~100x costlier than scVI's 30-dim one (one
        99-perm block took 2 h serially). Permutations are drawn eagerly in the main
        process so the sequence stays reproducible regardless of worker scheduling.
        """
        if not args.n_perm:
            return np.array([0.5])
        perms = [rng.permutation(y) for _ in range(args.n_perm)]
        scores = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(cv_scores)(Z, yp, args.seed + 1 + i) for i, yp in enumerate(perms))
        return np.array([s[0] for s in scores])

    disc_rows = []
    for subset in ("all", "confident"):
        base = np.ones(len(proxy), bool) if subset == "all" else proxy["confident"].values
        for a, b in pairs:
            sel_a = np.flatnonzero(base & (proxy["gene_space_label"].values == a))
            sel_b = np.flatnonzero(base & (proxy["gene_space_label"].values == b))
            n = min(len(sel_a), len(sel_b), args.max_per_class)
            if n < args.cv_folds * 2:
                log(f"   SKIP {a} vs {b} [{subset}]: too few cells "
                    f"({len(sel_a)} / {len(sel_b)})")
                continue
            ia = rng.choice(sel_a, n, replace=False)
            ib = rng.choice(sel_b, n, replace=False)
            idx = np.concatenate([ia, ib])
            y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
            for name, emb in emb_arrays.items():
                Z = emb[idx]
                auc, f1 = cv_scores(Z, y, args.seed)
                null = null_aucs(Z, y)
                pval = (1 + int((null >= auc).sum())) / (len(null) + 1)
                disc_rows.append({
                    "arm": name, "subset": subset, "label_a": a, "label_b": b,
                    "n_per_class": n, "roc_auc": round(float(auc), 4),
                    "macro_f1": round(float(f1), 4),
                    "null_auc_mean": round(float(null.mean()), 4),
                    "null_auc_p95": round(float(np.percentile(null, 95)), 4),
                    "auc_minus_null": round(float(auc - null.mean()), 4),
                    "p_vs_shuffle": round(pval, 4),
                })
                log(f"   [{name}] {a} vs {b} [{subset}] n={n}/class: "
                    f"AUC {auc:.3f} (null {null.mean():.3f}) F1 {f1:.3f} p={pval:.3f}")
    disc = pd.DataFrame(disc_rows)
    disc.to_csv(outdir / "pairwise_discriminability.csv", index=False)
    log(f"   wrote pairwise_discriminability.csv ({len(disc)} rows)")

    # ---------- D. cluster dilution: where do proxy-labelled cells land? ----------
    if annotations:
        log("D. cluster dilution of target labels within the published Leiden partitions")
        dil_rows = []
        for name, path in annotations.items():
            ann = pd.read_parquet(path).reindex(cells)
            if "leiden" not in ann.columns:
                continue
            leiden = ann["leiden"].astype(str)
            sizes = leiden.value_counts()
            assigned = set(ann["annotation"].dropna().astype(str).unique())
            for t in targets:
                sel = proxy["gene_space_label"].values == t
                n_t = int(sel.sum())
                if n_t == 0:
                    dil_rows.append({"arm": name, "label": t, "n_proxy_cells": 0})
                    continue
                per_cluster = leiden[sel].value_counts()
                frac_within = (per_cluster / sizes.reindex(per_cluster.index)).sort_values(ascending=False)
                top_cluster = per_cluster.idxmax()
                share = per_cluster / n_t
                ent = float(-(share * np.log(share)).sum())
                dil_rows.append({
                    "arm": name, "label": t, "n_proxy_cells": n_t,
                    "n_clusters_containing": int((per_cluster > 0).sum()),
                    "n_clusters_total": int(len(sizes)),
                    # concentration of the proxy population across clusters
                    "frac_in_top_cluster": round(float(share.max()), 4),
                    "spread_entropy": round(ent, 3),
                    # the quantity that directly explains a never-assigned argmax:
                    # if no cluster is mostly this type, no centroid will match it
                    "max_within_cluster_frac": round(float(frac_within.iloc[0]), 4),
                    "max_within_cluster_id": str(frac_within.index[0]),
                    "label_assigned_by_transfer": t in assigned,
                })
        dil = pd.DataFrame(dil_rows)
        dil.to_csv(outdir / "cluster_dilution.csv", index=False)
        log(f"   wrote cluster_dilution.csv ({len(dil)} rows)")
        for r in dil.itertuples():
            if getattr(r, "n_proxy_cells", 0):
                log(f"     [{r.arm}] {r.label}: {r.n_proxy_cells} proxy cells over "
                    f"{r.n_clusters_containing}/{r.n_clusters_total} clusters | "
                    f"top cluster holds {r.frac_in_top_cluster:.1%} of them, where they are "
                    f"{r.max_within_cluster_frac:.1%} of that cluster | "
                    f"transfer assigned it: {r.label_assigned_by_transfer}")

    log("DONE. Reading the result: high AUC + low max_within_cluster_frac = the embedding "
        "separates these cells and the clustering/argmax discards it (fix the transfer). "
        "AUC at the shuffle null on the confident subset, while the other arm scores well "
        "on the same pair, is the only pattern here that supports an embedding-level claim.")


if __name__ == "__main__":
    main()
