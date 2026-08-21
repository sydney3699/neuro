"""Cell type taxonomy harmonization between master and per-region files.

Two mappings coexist, for different callers -- keep both behaviors verbatim:
  - `harmonize_master_celltypes` (notebook-era): maps the master H1/H2 taxonomy to
    the per-region classes and returns None for progenitor types (RG/IPC/EN-Mig),
    dropping them from scope.
  - `GW20_H2_TO_COMMON` (GW20<->GW34 persistence pipeline): maps GW20 H2 labels to
    the common 8-class space, keeping progenitors as an explicit "Other-progenitor"
    bucket (and treating EN-L2 as EN-IT-UL). The committed GW34 persistence results
    depend on this exact mapping -- do not change its semantics.
"""

import pandas as pd


UL_H2 = {'EN-IT-L2/3', 'EN-IT-L3/4', 'EN-IT-L4'}
DL_H2 = {'EN-IT-L4/5', 'EN-IT-L6'}


def harmonize_master_celltypes(adata):
    """
    Add a 'harmonized' column to the master MERFISH AnnData, mapping its
    fine-grained H1/H2 taxonomy to the coarser per-region taxonomy
    (EN-IT-UL, EN-IT-DL, EN-ET, IN, Astrocyte, EC, Glia).

    Cells whose types do not exist in the per-region file (RG, IPC, EN-Mig)
    receive None (they're outside the cortical plate / marginal zone scope
    of the harmonized comparison).
    """
    def _map(row):
        h1, h2 = row['H1_annotation'], row['H2_annotation']
        if h1 == 'EN-IT':
            if h2 in UL_H2:
                return 'EN-IT-UL'
            if h2 in DL_H2:
                return 'EN-IT-DL'
            return None
        if h1 == 'EN-ET':
            return 'EN-ET'
        if h1 == 'IN':
            return 'IN'
        if h1 == 'EC':
            return 'EC'
        if (h1 == 'Glia' and h2 == 'Astro-1') or (h1 == 'RG' and h2 == 'Astro-late1'):
            return 'Astrocyte'
        if h1 == 'Glia':  # remaining Glia (OPC)
            return 'Glia'
        return None  # RG (non-Astro), IPC, EN-Mig

    adata.obs['harmonized'] = adata.obs.apply(_map, axis=1)
    return adata


# ---- GW20<->GW34 persistence pipeline: common 8-class space + GW20 H2 mapping ----
COMMON = ["EN-ET", "EN-IT-DL", "EN-IT-UL", "IN", "EC", "Astrocyte", "Glia", "Other-progenitor"]

# GW20 H2_annotation -> common class. Covers the union of H2 values seen in
# commot_v1/v2 (33 types). L4/5 assigned to DL (deep boundary; approximate, flagged).
GW20_H2_TO_COMMON = {
    # EN-IT upper-layer
    "EN-IT-L2/3": "EN-IT-UL", "EN-IT-L3/4": "EN-IT-UL", "EN-IT-L4": "EN-IT-UL", "EN-L2": "EN-IT-UL",
    # EN-IT deep-layer
    "EN-IT-L4/5": "EN-IT-DL", "EN-IT-L6": "EN-IT-DL",
    # EN-ET (incl. subplate — mature ET at GW20)
    "EN-ET-L5/6": "EN-ET", "EN-ET-SP": "EN-ET", "EN-ET-SP-P": "EN-ET",
    "EN-ET-SP-early": "EN-ET", "EN-ET-L6-early": "EN-ET",
    # interneurons (mature)
    "IN-MGE": "IN", "IN-SST": "IN", "IN-CGE": "IN",
    # endothelial
    "EC": "EC",
    # astrocytes / oligo-lineage
    "Astro-1": "Astrocyte", "Astro-late1": "Astrocyte", "OPC": "Glia",
    # progenitor / migrating / germinal-zone -> Other-progenitor (GW20-specific)
    "EN-IZ-1": "Other-progenitor", "EN-IZ-2": "Other-progenitor",
    "EN-oSVZ-1": "Other-progenitor", "En-oSVZ-2": "Other-progenitor",
    "INP-VZ/GE": "Other-progenitor", "IN-VZ/GE": "Other-progenitor",
    "RG1": "Other-progenitor", "oRG1": "Other-progenitor", "vRG-late": "Other-progenitor",
    "tRG": "Other-progenitor",
    "IPC-SVZ-1": "Other-progenitor", "IPC-SVZ-2": "Other-progenitor",
    "IPC-iSVZ": "Other-progenitor", "IPC-VZ/SVZ": "Other-progenitor",
}