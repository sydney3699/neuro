"""Data loading utilities for the V1/V2 spatial CCC project."""

from pathlib import Path
import anndata as ad
import pandas as pd


def get_project_root():
    """Find the project root by walking up from CWD until we find pyproject.toml."""
    p = Path.cwd()
    while p != p.parent:
        if (p / 'pyproject.toml').exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        "Could not find project root. Make sure you're inside the project directory."
    )


def load_main_merfish(project_root=None, harmonized=True, backed='r'):
    """
    Load the master MERFISH AnnData with optional harmonized celltype column.

    Note: Calls obs_names_make_unique() automatically because the master file
    contains duplicate cell IDs across sections.

    Parameters
    ----------
    project_root : Path or None
        Path to the project root directory. If None, auto-detect from CWD.
    harmonized : bool, default True
        If True, merge the saved 'harmonized' column into adata.obs.
        Set False if you haven't run the harmonization step yet.
    backed : str or None, default 'r'
        Backing mode passed to read_h5ad. Use 'r' to save memory; None for full load.

    Returns
    -------
    AnnData
    """
    if project_root is None:
        project_root = get_project_root()

    raw_path = project_root / 'data' / 'raw' / 'merscope_integrated_855.h5ad'
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Master MERFISH file not found at {raw_path}. "
            "Download from Zenodo (10.5281/zenodo.14422018) and place in data/raw/."
        )

    adata = ad.read_h5ad(raw_path, backed=backed)
    
    # Cell IDs are not unique across sections; make them unique deterministically
    adata.obs_names_make_unique()

    if harmonized:
        harmonized_csv = project_root / 'data' / 'processed' / 'main_merfish_obs_harmonized.csv'
        if not harmonized_csv.exists():
            raise FileNotFoundError(
                f"Harmonized obs file not found at {harmonized_csv}. "
                "Run the harmonization step in 01_data_inspection.ipynb first, "
                "or call this function with harmonized=False."
            )
        harmonized_df = pd.read_csv(harmonized_csv, index_col=0)
        
        # Defensive check: lengths must match
        if len(harmonized_df) != adata.n_obs:
            raise ValueError(
                f"Harmonized CSV has {len(harmonized_df)} rows but AnnData has {adata.n_obs} cells. "
                "The harmonization CSV is out of date — re-run the harmonization step."
            )
        
        # Use positional assignment to avoid index-based merging issues
        adata.obs['harmonized'] = harmonized_df['harmonized'].values

    return adata