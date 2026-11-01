#!/usr/bin/env python
"""
Precompute snRNA reference centroids over the shared gene set, as a small CSV.

The cluster->reference label transfer only ever consumes per-type MEAN log1p-CP10k
profiles over the genes shared between the imputation panel and the reference -- a
table of roughly (n_types x 300) floats. Carrying the multi-GB reference h5ad to
wherever the analysis runs is therefore unnecessary, and sometimes impossible: the
Explorer scratch copy was purged, and re-uploading 1.8 GB over a flaky link is far
slower than shipping the ~1 MB table the transfer actually needs.

This writes that table so embedding_discriminability.py --ref-centroids-csv can
consume it. Normalization is neurospatial.expression.lognorm_mean_by -- the same
helper the probe uses -- so the CSV is identical to what the probe would have
computed from the h5ad itself.

Gene order: `shared` follows the imputation-gene order (as in annotate_cells.py),
intersected with the reference's var_names. Pass --imputation-genes as a newline-
delimited file to avoid loading the spatial h5ad just for its uns field; extract it
cheaply with h5py: f["uns/imputation_genes"][:].

Runs anywhere numpy/pandas/anndata are installed (no scanpy, no GPU).
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from neurospatial.expression import lognorm_mean_by


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Precompute reference centroids over shared genes")
    p.add_argument("--reference-h5ad", required=True)
    p.add_argument("--ref-label-key", default="celltypes")
    p.add_argument("--imputation-genes", required=True,
                   help="newline-delimited imputation gene list (defines the gene order)")
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    import anndata as ad

    genes = [g.strip() for g in Path(args.imputation_genes).read_text().split("\n") if g.strip()]
    log(f"{len(genes)} imputation genes")

    log(f"loading reference {args.reference_h5ad}")
    ref = ad.read_h5ad(args.reference_h5ad)
    if args.ref_label_key not in ref.obs:
        raise SystemExit(f"reference has no obs['{args.ref_label_key}']; "
                         f"have {list(ref.obs.columns)}")
    ref_vars = [str(v) for v in ref.var_names]
    shared = [g for g in genes if g in set(ref_vars)]
    log(f"  {ref.n_obs} reference cells; {len(shared)} shared genes (of {len(genes)})")

    labels = ref.obs[args.ref_label_key].astype(str).values
    cent = lognorm_mean_by(ref.X, ref_vars, labels, shared)
    cent.index.name = args.ref_label_key
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cent.to_csv(out)
    log(f"wrote {out} ({cent.shape[0]} types x {cent.shape[1]} genes, "
        f"{out.stat().st_size / 1e6:.2f} MB)")
    log(f"  types: {', '.join(map(str, cent.index[:8]))}"
        f"{' ...' if len(cent) > 8 else ''}")


if __name__ == "__main__":
    main()
