#!/usr/bin/env python
"""
Collapse the 35 fine snRNA `celltypes` labels into ~16 coarse groups, written as a
new obs column `coarse_celltypes` on a COPY of the reference (snrna.h5ad is never
mutated in place). Used to test whether the scVI(37 cluster)-vs-scGPT(14 cluster)
niche-composition gap (Q1 in compare_niches.py) is a granularity artifact: with
fewer, more separable reference classes, cluster-count differences between
embeddings should matter less to the resulting composition.

Fine -> coarse groups (16): EN-ET-1..7 -> EN-ET; EN-IT-DL-1..3 -> EN-IT-DL;
EN-IT-UL-1..5 -> EN-IT-UL; IN-CGE-1..2 -> IN-CGE; IN-MGE-1..3 -> IN-MGE;
RG + vRG -> RG; EC-1 + EC-2 -> EC; EN-Mig-1..3 -> EN-Mig (transient migrating
excitatory state, kept distinct from settled EN types). EN-IT-L4-V1 and EN-L2 are
kept as their own singleton groups rather than merged into EN-IT-UL/DL: L4-V1 is
the V1-specific layer marker central to this project's V1-vs-V2 question, and
EN-L2 is a distinct upper-layer identity. IPC, OPC, Microglia, INP-CGE,
Dividing progenitor, unknown are kept as singletons (each already a distinct,
non-redundant class).
"""
import argparse

FINE_TO_COARSE = {
    "EN-ET-1": "EN-ET", "EN-ET-2": "EN-ET", "EN-ET-3": "EN-ET", "EN-ET-4": "EN-ET",
    "EN-ET-5": "EN-ET", "EN-ET-6": "EN-ET", "EN-ET-7": "EN-ET",
    "EN-IT-DL-1": "EN-IT-DL", "EN-IT-DL-2": "EN-IT-DL", "EN-IT-DL-3": "EN-IT-DL",
    "EN-IT-UL-1": "EN-IT-UL", "EN-IT-UL-2": "EN-IT-UL", "EN-IT-UL-3": "EN-IT-UL",
    "EN-IT-UL-4": "EN-IT-UL", "EN-IT-UL-5": "EN-IT-UL",
    "EN-IT-L4-V1": "EN-IT-L4-V1",
    "EN-L2": "EN-L2",
    "EN-Mig-1": "EN-Mig", "EN-Mig-2": "EN-Mig", "EN-Mig-3": "EN-Mig",
    "IN-CGE-1": "IN-CGE", "IN-CGE-2": "IN-CGE",
    "IN-MGE-1": "IN-MGE", "IN-MGE-2": "IN-MGE", "IN-MGE-3": "IN-MGE",
    "INP-CGE": "INP-CGE",
    "RG": "RG", "vRG": "RG",
    "EC-1": "EC", "EC-2": "EC",
    "IPC": "IPC",
    "OPC": "OPC",
    "Microglia": "Microglia",
    "Dividing progenitor": "Dividing progenitor",
    "unknown": "unknown",
}


def main():
    p = argparse.ArgumentParser(description="Add obs['coarse_celltypes'] to a copy of the snRNA reference")
    p.add_argument("--reference-h5ad", default="/scratch/cole.sy/neuro/data/raw/snrna.h5ad")
    p.add_argument("--fine-key", default="celltypes")
    p.add_argument("--out", default="/scratch/cole.sy/neuro/data/raw/snrna_coarse.h5ad")
    args = p.parse_args()

    import anndata as ad

    ref = ad.read_h5ad(args.reference_h5ad)
    fine = ref.obs[args.fine_key].astype(str)
    missing = sorted(set(fine.unique()) - set(FINE_TO_COARSE))
    if missing:
        raise SystemExit(f"no coarse mapping for fine labels: {missing}")
    ref.obs["coarse_celltypes"] = fine.map(FINE_TO_COARSE).astype("category")

    n_fine = fine.nunique()
    n_coarse = ref.obs["coarse_celltypes"].nunique()
    print(f"{n_fine} fine celltypes -> {n_coarse} coarse groups")
    print(ref.obs["coarse_celltypes"].value_counts())

    ref.write_h5ad(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
