import itertools
from multiprocessing import Pool
from typing import Any, List, Tuple, Union

import anndata
import mopsy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_array, csr_matrix

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def remap_anndata(
    h5ad_or_adata: Union[str, anndata.AnnData],
    feature_set_order: dict,
    var_feature_column: str = "index",
    layer_matrix_name: str = "counts",
    consolidate_duplicate_gene_func=sum,
) -> csr_matrix:
    """Extract and remap the count matrix to the provided feature (gene) set order from the :py:class:`~anndata.AnnData`
    object.

    Args:
        adata:
            Input ``AnnData`` object.

            Alternatively, may also provide a path to the H5ad file.

            The index of the `var` slot must contain the feature ids
            for the columns in the matrix.

        feature_set_order:
            A dictionary with the feature ids as keys and their index as
            value (e.g. gene symbols). The feature ids from the
            ``AnnData`` object are remapped to the feature order from
            this dictionary.

        var_feature_column:
            Column in ``var`` containing the feature ids (e.g. gene symbols).
            Defaults to the index of the ``var`` slot.

        layer_matrix_name:
            Layer containing the matrix to add to TileDB.
            Defaults to "counts".

        consolidate_duplicate_gene_func:
            Function to consolidate when the AnnData object contains
            multiple rows with the same feature id or gene symbol.

            Defaults to :py:func:`sum`.

    Returns:
        A ``csr_matrix`` representation of the assay matrix.
    """

    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata, "r")
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    omat = adata.layers[layer_matrix_name]

    if len(feature_set_order) == 0:
        return coo_matrix(
            ([], ([], [])),
            shape=(adata.shape[0], len(feature_set_order)),
        ).tocsr()

    if var_feature_column == "index":
        osymbols = adata.var.index.tolist()
    else:
        osymbols = adata.var[var_feature_column].tolist()

    mat, symbols = consolidate_duplicate_symbols(
        omat,
        feature_ids=osymbols,
        consolidate_duplicate_gene_func=consolidate_duplicate_gene_func,
    )

    # figure out which indices to keep from the matrix
    indices_to_keep = [i for i, x in enumerate(symbols) if x in feature_set_order]
    symbols_to_keep = [symbols[i] for i in indices_to_keep]

    mat = mat[:, indices_to_keep].copy()

    if len(indices_to_keep) == 0:
        return coo_matrix(
            ([], ([], [])),
            shape=(adata.shape[0], len(feature_set_order)),
        ).tocsr()

    # figure out mapping from the current indices to the original feature order
    indices_to_map = []
    for x in symbols_to_keep:
        indices_to_map.append(feature_set_order[x])

    if isinstance(mat, np.ndarray):
        mat_coo = coo_matrix(mat)
    elif isinstance(mat, (csr_array, csr_matrix)):
        mat_coo = mat.tocoo()
    else:
        raise TypeError(f"Unknown matrix type: {type(mat)}.")

    # remap gene symbols to the new feature order
    new_col = np.array([indices_to_map[i] for i in mat_coo.col])

    return coo_matrix(
        (mat_coo.data, (mat_coo.row, new_col)),
        shape=(adata.shape[0], len(feature_set_order)),
    ).tocsr()


def consolidate_duplicate_symbols(
    matrix: Any, feature_ids: List[str], consolidate_duplicate_gene_func: callable
) -> anndata.AnnData:
    """Consolidate duplicate gene symbols.

    Args:
        matrix:
            data matrix with rows
            for cells and columns for genes.

        feature_ids:
            List of feature ids along the column axis of the matrix.

        consolidate_duplicate_gene_func:
            Function to consolidate when the AnnData object contains
            multiple rows with the same feature id or gene symbol.

            Defaults to :py:func:`sum`.

    Returns:
        AnnData object with duplicate gene symbols consolidated.
    """

    if len(set(feature_ids)) == len(feature_ids):
        return matrix, feature_ids

    return mopsy.apply(
        consolidate_duplicate_gene_func, mat=matrix, group=feature_ids, axis=1
    )


def _extract_info(
    h5ad_or_adata: Union[str, anndata.AnnData],
    var_feature_column: str = "index",
    obs_subset_columns: List[str] = None,
) -> Tuple[List[str], pd.DataFrame, int]:
    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata, "r")
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    if var_feature_column == "index":
        symbols = adata.var.index.tolist()
    else:
        symbols = adata.var[var_feature_column].tolist()

    features = pd.DataFrame({})
    if obs_subset_columns is not None and len(obs_subset_columns) > 0:
        _common = list(set(obs_subset_columns).intersection(adata.obs))

        if len(_common) > 0:
            features = adata.obs[_common]

        for col in obs_subset_columns:
            if col not in features.columns:
                features[col] = ["NA"] * adata.shape[0]

    return symbols, features, adata.shape[0]


def _wrapper_extract_info(args):
    file, gcol, ccols = args
    return _extract_info(file, gcol, ccols)


def extract_anndata_info(
    h5ad_or_adata: List[Union[str, anndata.AnnData]],
    var_feature_column: str = "index",
    obs_subset_columns: dict = None,
    num_threads: int = 1,
):
    """Extract and generate the list of unique feature identifiers and cell counts across files.

    Args:
        h5ad_or_adata:
            List of anndata objects or path to h5ad files.

        var_feature_column:
            Column containing the feature ids (e.g. gene symbols).
            Defaults to "index".

        obs_subset_columns:
            List of obs columns to concatenate across all files.
            Defaults to None and no metadata columns will be extracted.

        num_threads:
            Number of threads to use.
            Defaults to 1.

        force:
            Whether to rescan all the files even though the cache exists.
            Defaults to False.
    """
    with Pool(num_threads) as p:
        _args = [
            (file_info, var_feature_column, obs_subset_columns)
            for file_info in h5ad_or_adata
        ]
        return p.map(_wrapper_extract_info, _args)


def scan_for_features(cache, unique: bool = True) -> List[str]:
    """Extract and generate the list of unique feature identifiers across files.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

        unique:
            Compute gene list to a unique list.

    Returns:
        List of all unique feature ids across all files.
    """
    _features = [x[0] for x in cache]
    if unique:
        return list(set(itertools.chain.from_iterable(_features)))

    return _features


def scan_for_cellcounts(cache) -> List[int]:
    """Extract cell counts across files.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

    Returns:
        List of cell counts across files.
    """

    _cellcounts = [x[2] for x in cache]
    return _cellcounts


def scan_for_cellmetadata(cache) -> List[int]:
    """Extract and merge all cell metadata data frames across files.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

    Returns:
        A :py:class:`pandas.Dataframe` containing all cell metadata.
    """

    _cellmeta = pd.concat([x[1] for x in cache])
    return _cellmeta
