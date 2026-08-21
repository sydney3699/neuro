#!/usr/bin/env python
"""
Ground-truth spatial-method evaluation: how well does each spatial-domain method
(Banksy vs STAGATE) recover cortical LAYER structure, per region across the k-sweep?

This is the reframed methodological-Q2 evaluation. The brief originally asked for
V1/V2 AREAL boundary recovery, but in FB080-O1b V1 and V2 are physically separate
dissection pieces (~1.7 mm apart) -- there is no contiguous areal boundary in the
imaged tissue to recover, and any spatial method trivially separates two pieces that
far apart. Cortical LAYERS, by contrast, are contiguous with real transitions and
have ground-truth labels (obs['layer']: mz/l2..l6/sp), so layer recovery is the
meaningful "does the GNN spatial method recover known structure better?" test.

Ground truth = obs['layer']. Predicted = domain_k{k} from the domain parquet. Per
(region, method, k):
  - ari_vs_layer       : adjusted_rand_score(layer, domain)  -- global agreement
  - v_measure          : homogeneity+completeness harmonic mean
  - layer_purity       : size-weighted purity, sum_d max_l count(d,l) / N
                         (same definition as {region}_{method}_ksweep_summary.csv --
                          cross-checkable)
  - layer_bnd_recall   : of true layer-boundary cells, fraction also domain-boundary
  - domain_bnd_precision: of domain-boundary cells, fraction at a true layer boundary
  - boundary_f1        : harmonic mean of the two (boundary "sharpness")
A cell is a layer/domain boundary cell if >= bnd_frac of its kNN sit in a different
layer/domain (kNN graph on spatial coords, built once per region).

Runs in the neuro env (anndata/pandas/numpy/scikit-learn).
"""
import argparse
import time
from pathlib import Path

from neurospatial.metrics import size_weighted_purity, boundary_metrics


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Layer-recovery ground-truth eval for spatial-domain methods")
    p.add_argument("--results", default="/scratch/cole.sy/neuro/results")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--methods", default="banksy,stagate")
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=14)
    p.add_argument("--layer-key", default="layer")
    p.add_argument("--knn", type=int, default=15, help="neighbors for the boundary graph")
    p.add_argument("--bnd-frac", type=float, default=0.5,
                   help="cell is a boundary cell if >= this fraction of kNN are in a different label")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, v_measure_score
    from sklearn.neighbors import NearestNeighbors

    results = Path(args.results)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = args.regions.split(",")
    methods = args.methods.split(",")
    ks = list(range(args.kmin, args.kmax + 1))

    rows = []
    for region in regions:
        h5ad = results / "commot" / f"commot_{region}.h5ad"
        log(f"Loading {h5ad}")
        a = ad.read_h5ad(h5ad)
        obs = a.obs
        layer = obs[args.layer_key].astype(str)
        truth_codes = layer.astype("category").cat.codes.to_numpy()
        xy = np.asarray(a.obsm["spatial"], dtype=float)
        # kNN boundary graph -- built ONCE per region, shared across methods/k
        log(f"  {a.n_obs} cells; building kNN (k={args.knn}) boundary graph")
        nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(xy)
        _, idx = nn.kneighbors(xy)
        knn_idx = idx[:, 1:]                                    # drop self

        for method in methods:
            pq = results / method / f"{region}_{method}_domains.parquet"
            if not pq.exists():
                log(f"  SKIP {region}/{method}: {pq} not found")
                continue
            dom_df = pd.read_parquet(pq)
            dom_df = dom_df.reindex(obs.index)                 # align to commot cells by id
            for k in ks:
                col = f"domain_k{k}"
                if col not in dom_df.columns or dom_df[col].isna().any():
                    log(f"  SKIP {region}/{method} {col}: missing/unaligned")
                    continue
                pred = dom_df[col].astype(str)
                pred_codes = pred.astype("category").cat.codes.to_numpy()
                ari = adjusted_rand_score(layer.values, pred.values)
                vmeas = v_measure_score(layer.values, pred.values)
                purity = size_weighted_purity(pred.values, layer.values)
                recall, precision, f1 = boundary_metrics(pred_codes, truth_codes, knn_idx, args.bnd_frac)
                rows.append({
                    "region": region, "method": method, "k": k,
                    "n_domains": int(pred.nunique()),
                    "ari_vs_layer": round(ari, 4), "v_measure": round(vmeas, 4),
                    "layer_purity": round(purity, 4),
                    "layer_bnd_recall": round(recall, 4), "domain_bnd_precision": round(precision, 4),
                    "boundary_f1": round(f1, 4),
                })
            log(f"  {region}/{method}: scored k={ks[0]}..{ks[-1]}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "layer_recovery_by_k.csv", index=False)
    log(f"wrote layer_recovery_by_k.csv ({len(df)} rows)")
    # headline: per region x method mean over k, and the k=8 slice
    if len(df):
        k8 = df[df["k"] == 8][["region", "method", "ari_vs_layer", "v_measure", "layer_purity", "boundary_f1"]]
        log("k=8 slice:\n" + k8.to_string(index=False))
        agg = df.groupby(["region", "method"])[["ari_vs_layer", "v_measure", "layer_purity", "boundary_f1"]].mean().round(3)
        log("mean over k=6..14:\n" + agg.to_string())


if __name__ == "__main__":
    main()
