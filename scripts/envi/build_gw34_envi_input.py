#!/usr/bin/env python
"""GW34 Stage 0 — build the GW34 ENVI input h5ad, mirroring the GW20 FB080 input schema.

Subset GW34 BA17 to the area-labeled V1/V2 cells (A-V1/B-V2, all of which are the
cp/mz layers), set X = raw integer counts (GW34's .X is normalized/log; the raw
counts live in .raw — ENVI needs counts), and keep obsm['spatial'] + the obs
annotations. Output feeds run_envi.py exactly like FB080_O1b_v1v2_envi_input.h5ad.
Panel gene symbols are kept as-is (run_envi.py handles HGNC aliases downstream).
Runs in the neuro env.
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def main():
    p = argparse.ArgumentParser(description="Build GW34 ENVI input (area-labeled, raw counts)")
    p.add_argument("--in-h5ad", default="/scratch/cole.sy/neuro/data/raw/gw34_umb5900_ba17.h5ad")
    p.add_argument("--out-h5ad", default="/scratch/cole.sy/neuro/data/processed/gw34_ba17_v1v2_envi_input.h5ad")
    p.add_argument("--areas", default="A-V1,B-V2")
    args = p.parse_args()

    a = ad.read_h5ad(args.in_h5ad)
    print(f"loaded {a.shape}  obs cols: {list(a.obs.columns)}  has raw: {a.raw is not None}", flush=True)

    areas = args.areas.split(",")
    a = a[a.obs["area"].isin(areas).values].copy()
    print(f"after area subset {areas}: {a.shape}", flush=True)
    print("area:\n" + a.obs["area"].value_counts(dropna=False).to_string(), flush=True)
    print("layer:\n" + a.obs["layer"].value_counts(dropna=False).to_string(), flush=True)

    if a.raw is None:
        raise SystemExit("no .raw present; cannot recover raw counts")
    raw = a.raw.to_adata()
    print(f"raw shape {raw.shape}; raw var sample: {list(map(str, raw.var_names[:5]))}", flush=True)

    # align raw to the 300-gene panel var order (panel symbols kept as-is)
    raw_set = set(map(str, raw.var_names))
    missing = [g for g in map(str, a.var_names) if g not in raw_set]
    if missing:
        print(f"WARNING: {len(missing)} panel genes absent from raw (kept from .X-space): {missing[:10]}", flush=True)
    common = [g for g in map(str, a.var_names) if g in raw_set]
    raw = raw[:, common]
    Xraw = raw.X
    arr = Xraw.toarray() if sp.issparse(Xraw) else np.asarray(Xraw)
    frac_int = float(np.mean(arr == np.round(arr)))
    print(f"raw X: shape {arr.shape}, integer-valued fraction={frac_int:.4f}, max={float(arr.max()):.1f}", flush=True)
    if frac_int < 0.99:
        print("WARNING: raw X does not look like integer counts — check the source .raw", flush=True)

    keep_obs = [c for c in ["gw", "sample", "region", "area", "layer",
                            "H1_annotation", "H2_annotation", "H3_annotation", "cortical_depth"]
                if c in a.obs.columns]
    out = ad.AnnData(X=sp.csr_matrix(arr), obs=a.obs[keep_obs].copy(), var=raw.var.copy())
    out.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)

    Path(args.out_h5ad).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    print(f"\nwrote {out.shape} -> {args.out_h5ad}", flush=True)
    print(f"  genes={out.n_vars}  spatial={out.obsm['spatial'].shape}  X integer-frac={frac_int:.3f}", flush=True)
    print(f"  obs cols: {list(out.obs.columns)}", flush=True)


if __name__ == "__main__":
    main()
