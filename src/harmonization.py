"""Cell type taxonomy harmonization between master and per-region files."""

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