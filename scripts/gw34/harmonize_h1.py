#!/usr/bin/env python
"""GW34 Stage 4 — harmonize GW20 and GW34 cell-type labels to a common class space
for the cross-timepoint composition / niche persistence comparison.

The two timepoints share only 5/31 H2 types (different developmental maturity), so
comparison is done at a common broad-class space:

    EN-ET, EN-IT-DL, EN-IT-UL, IN, EC, Astrocyte, Glia, Other-progenitor

- **GW34** already carries native `H1_annotation` in exactly this space
  {Astrocyte, EC, EN-ET, EN-IT-DL, EN-IT-UL, Glia, IN} → used directly.
- **GW20** has a single `EN-IT` H1 and lumps astro/OPC under `Glia`, and has ~15%
  progenitor/migrating cells absent at GW34. So GW20 is mapped from its FINER
  `H2_annotation` via the explicit dict below (per user decisions 2026-08-19):
  EN-IT split into DL/UL from H2 layer labels; progenitor/migrating/germinal types
  (EN-Mig/RG/IPC/oSVZ/IZ/VZ-GE) collapsed to `Other-progenitor` (kept, not dropped,
  so composition fractions stay honest and the GW20-only maturation signal shows).

Writes a per-cell annotation parquet per (stage, region), keyed by cell id with a
single `annotation` column, consumable by profile_niches.py --annotation-parquet.
Runs in the neuro env.
"""
import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from neurospatial.harmonization import COMMON, GW20_H2_TO_COMMON

def log(m):
    print(m, flush=True)


def main():
    p = argparse.ArgumentParser(description="Harmonize GW20/GW34 labels to a common class space")
    p.add_argument("--gw20-commot-dir", default="/scratch/cole.sy/neuro/results/commot")
    p.add_argument("--gw34-commot-dir", default="/scratch/cole.sy/neuro/results/commot_gw34")
    p.add_argument("--regions", default="v1,v2")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = args.regions.split(",")
    prov = []

    # ---- GW20: map from H2_annotation ----
    for r in regions:
        h5 = Path(args.gw20_commot_dir) / f"commot_{r}.h5ad"
        if not h5.exists():
            log(f"[gw20 {r}] SKIP: {h5} not found"); continue
        obs = ad.read_h5ad(h5, backed="r").obs
        h2 = obs["H2_annotation"].astype(str)
        unmapped = sorted(set(h2.unique()) - set(GW20_H2_TO_COMMON))
        if unmapped:
            log(f"[gw20 {r}] WARNING unmapped H2 -> 'Other-progenitor': {unmapped}")
        harm = h2.map(GW20_H2_TO_COMMON).fillna("Other-progenitor")
        df = pd.DataFrame({"annotation": harm.values}, index=obs.index)
        out = outdir / f"gw20_{r}_h1harm.parquet"
        df.to_parquet(out)
        vc = harm.value_counts()
        log(f"[gw20 {r}] {len(df)} cells -> {out.name}\n{vc.to_string()}\n")
        for cls, n in vc.items():
            prov.append({"stage": "gw20", "region": r, "class": cls, "n": int(n)})

    # ---- GW34: native H1_annotation (already in the common space) ----
    for r in regions:
        h5 = Path(args.gw34_commot_dir) / f"commot_{r}.h5ad"
        if not h5.exists():
            log(f"[gw34 {r}] SKIP: {h5} not found (run Stage 2 first)"); continue
        obs = ad.read_h5ad(h5, backed="r").obs
        h1 = obs["H1_annotation"].astype(str)
        off = sorted(set(h1.unique()) - set(COMMON))
        if off:
            log(f"[gw34 {r}] NOTE H1 values outside common space (kept as-is): {off}")
        df = pd.DataFrame({"annotation": h1.values}, index=obs.index)
        out = outdir / f"gw34_{r}_h1harm.parquet"
        df.to_parquet(out)
        vc = h1.value_counts()
        log(f"[gw34 {r}] {len(df)} cells -> {out.name}\n{vc.to_string()}\n")
        for cls, n in vc.items():
            prov.append({"stage": "gw34", "region": r, "class": cls, "n": int(n)})

    pd.DataFrame(prov).to_csv(outdir / "h1harm_class_counts.csv", index=False)
    log(f"wrote per-(stage,region) harmonized parquets + h1harm_class_counts.csv to {outdir}")


if __name__ == "__main__":
    main()
