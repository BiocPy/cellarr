from typing import Union

import anndata
import numpy as np
from scipy.sparse import coo_matrix, csr_array, csr_matrix

__author__ = "Jayaram Kancherla, Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def remap_anndata(
    h5ad_or_adata: Union[str, anndata.AnnData],
    gene_set_order: dict,
    var_gene_column: str = "index",
    count_layer_name: str = "counts",
) -> coo_matrix:
    """Extract and remap the count matrix to the provided gene set order
    from the :py:class:`~anndata.AnnData` object.

    Args:
        adata:
            Input ``AnnData`` object.

            Alternatively, may also provide a path to the H5ad file.

            The index of the `var` slot must contain the gene symbols
            for the columns in the matrix.

        gene_set_order:
            A dictionary with the symbols as keys and their index as value.
            The gene symbols from the ``AnnData`` object are remapped to the
            gene order from this dictionary.

        var_gene_column:
            Column in ``var`` containing the symbols.
            Defaults to the index of the ``var`` slot.

        count_layer_name:
            Layer containing the matrix to add to tiledb.
            Defaults to "counts".

    Returns:
        A ``csr_matrix`` representation of the assay matrix.
    """

    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata)
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    mat = adata.layers[count_layer_name]

    if var_gene_column == "index":
        symbols = adata.var.index.tolist()
    else:
        symbols = adata.var[var_gene_column].tolist()

    indices_to_keep = [i for i, x in enumerate(symbols) if x in gene_set_order]
    symbols_to_keep = [symbols[i] for i in indices_to_keep]

    mat = mat[:, indices_to_keep].copy()

    indices_to_map = []
    for x in symbols_to_keep:
        indices_to_map.append(gene_set_order[x])

    if isinstance(mat, np.ndarray):
        mat_coo = coo_matrix(mat)
    elif isinstance(mat, (csr_array, csr_matrix)):
        mat_coo = mat.tocoo()
    else:
        raise TypeError(f"Unknown matrix type: {type(mat)}.")

    new_col = np.array([indices_to_map[i] for i in mat_coo.col])

    return coo_matrix(
        (mat_coo.data, (mat_coo.row, new_col)),
        shape=(adata.shape[0], len(gene_set_order)),
    ).tocsr()
