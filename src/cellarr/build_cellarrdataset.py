"""Build the `CellArrDatset`

The `CellArrDataset` method is designed to store single-cell RNA-seq
datasets but can be generalized to store any 2-dimensional experimental data.

This method creates three TileDB files in the directory specified by `output_path`:
- `gene_metadata`: A TileDB file containing gene metadata.
- `cell_metadata`: A TileDB file containing cell metadata.
- A matrix TileDB file named as specified by the `layer_matrix_name` parameter.

The TileDB matrix file is stored in a cell X gene orientation. This orientation
is chosen because the fastest-changing dimension as new files are added to the
collection is usually the cells rather than genes.

## Process

1. **Scan the Collection**: Scan the entire collection of files to create
a unique set of gene symbols. Store this gene set as the
`gene_metadata` TileDB file.
2. **Store Cell Metadata**: Store cell metadata as the
`cell_metadata` TileDB file.
3. **Remap and Orient Data**: For each dataset in the collection,
remap and orient the gene dimension using the gene set from Step 1.
This step ensures consistency in gene measurement and order, even if
some genes are unmeasured or ordered differently in the original experiments.


Example:

    .. code-block:: python

        import anndata
        import numpy as np
        import tempfile
        from cellarr import build_cellarrdataset, CellArrDataset

        # Create a temporary directory
        tempdir = tempfile.mkdtemp()

        # Read AnnData objects
        adata1 = anndata.read_h5ad("path/to/object1.h5ad")
        # or just provide the path
        adata2 = "path/to/object2.h5ad"

        # Build CellArrDataset
        dataset = build_cellarrdataset(
            output_path=tempdir,
            h5ad_or_adata=[adata1, adata2],
            matrix_dim_dtype=np.float32
        )
"""

import os
import warnings
from typing import List, Union

import anndata
import numpy as np
import pandas as pd

from . import utils_anndata as uad
from . import utils_tiledb_array as uta
from . import utils_tiledb_frame as utf
from .CellArrDataset import CellArrDataset

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def build_cellarrdataset(
    files: List[Union[str, anndata.AnnData]],
    output_path: str,
    num_cells: int = None,
    num_genes: int = None,
    cell_metadata: Union[pd.DataFrame, str] = None,
    gene_metadata: Union[List[str], dict, str, pd.DataFrame] = None,
    var_gene_column: str = "index",
    layer_matrix_name: str = "counts",
    skip_gene_tiledb: bool = False,
    skip_cell_tiledb: bool = False,
    skip_matrix_tiledb: bool = False,
    cell_dim_dtype: np.dtype = np.uint32,
    gene_dim_dtype: np.dtype = np.uint32,
    matrix_dim_dtype: np.dtype = np.uint32,
    optimize_tiledb: bool = True,
    num_threads: int = 1,
):
    """Generate the `CellArrDataset`.

    All files are expected to be consistent and any modifications
    to make them consistent is outside the scope of this function
    and package.

    There's a few assumptions this process makes:
    - If object in ``files`` is an AnnData or H5AD object, these must
    contain an assay matrix in layer names as ``layer_matrix_name``
    parameter.
    - Feature information must contain a column defined by
    ``var_gene_column`` that contains gene symbols or a common entity
    across all files.
    -

    Args:
        files:
            List of file paths to `H5AD` or ``AnnData`` objects.
            Each object in this list must contain
            - gene symbols as index or the column specified by
            ``var_gene_column``.
            - Must contain a layers with a matrix named as
            ``layer_matrix_name``.

        output_path:
            Path to where the output tiledb files should be stored.

        num_cells:
            Number of cells across all files.

            Defualts to None, in which case, automatically inferred by
            scanning all objects in ``h5ad_or_adata``.

        num_genes:
            Number of genes across all cells.

            Defualts to None, in which case, automatically inferred by
            scanning all objects in ``h5ad_or_adata``.

        cell_metadata:
            Path to the file containing a concatenated cell metadata across
            all cells. In this case, the first row is expected to contain the
            column names.

            Alternatively, may also provide a dataframe containing the cell
            metadata across all objects.

            Regardless of the input type, the number of rows in the file or
            DataFrame must match the ``num_cells`` argument.

            Defaults to None, then a simple range index is created using the
            ``num_cells`` argument.

        gene_metadata:
            Path to the file containing a concatenated gene annotations across
            all datasets. In this case, the first row is
            expected to contain the column names and an index column
            containing the gene symbols to remap the matrix.

            Alternatively, may also provide a dataframe containing the gene
            annotations across all objects.

            Alternatively, a list or a dictionary of gene symbols.

            Regardless of the input type, the number of rows in the file or
            DataFrame must match the ``num_genes`` argument.

            Defaults to None, then a gene set is generated by scanning all
            objects in ``h5ad_or_adata``.

        var_gene_column:
            Column name from ``var`` slot that contains the gene symbols.
            Must be consistent across all objects in ``h5ad_or_adata``.

            Defaults to "index".

        layer_matrix_name:
            Matrix name from ``layers`` slot to add to tiledb.
            Must be consistent across all objects in ``h5ad_or_adata``.

            Defaults to "counts".

        skip_gene_tiledb:
            Whether to skip generating gene metadata tiledb.

            Defaults to False.

        skip_cell_tiledb:
            Whether to skip generating cell metadata tiledb.

            Defaults to False.

        skip_matrix_tiledb:
            Whether to skip generating matrix tiledb.

            Defaults to False.

        cell_dim_dtype:
            NumPy dtype for the cell dimension.
            Defaults to np.uint32.

            Note: make sure the number of cells fit
            within the range limits of unsigned-int32.

        gene_dim_dtype:
            NumPy dtype for the gene dimension.
            Defaults to np.uint32.

            Note: make sure the number of genes fit
            within the range limits of unsigned-int32.

        matrix_dim_dtype:
            NumPy dtype for the values in the matrix.
            Defaults to np.uint32.

            Note: make sure the matrix values fit
            within the range limits of unsigned-int32.

        num_threads:
            Number of threads.

            Defaults to 1.
    """
    if not os.path.isdir(output_path):
        raise ValueError("'output_path' must be a directory.")

    if gene_metadata is None:
        warnings.warn(
            "Scanning all files for gene symbols, this may take long", UserWarning
        )
        gene_set = uad.scan_for_genes(
            files, var_gene_column=var_gene_column, num_threads=num_threads
        )

        gene_set = sorted(gene_set)

        gene_metadata = pd.DataFrame({"genes": gene_set}, index=gene_set)
    elif isinstance(gene_metadata, list):
        _gene_list = sorted(list(set(gene_metadata)))
        gene_metadata = pd.DataFrame({"genes": _gene_list}, index=_gene_list)
    elif isinstance(gene_metadata, dict):
        _gene_list = sorted(list(gene_metadata.keys()))
        gene_metadata = pd.DataFrame({"genes": _gene_list}, index=_gene_list)
    elif isinstance(gene_metadata, str):
        gene_metadata = pd.read_csv(gene_metadata, index=True, header=True)

    gene_metadata["genes_index"] = gene_metadata.index.tolist()

    if not isinstance(gene_metadata, pd.DataFrame):
        raise TypeError("'gene_metadata' must be a pandas dataframe.")

    if len(gene_metadata.index.unique()) != len(gene_metadata.index.tolist()):
        raise ValueError("'gene_metadata' must contain a unique index.")

    if num_genes is None:
        num_genes = len(gene_metadata)

    # Create the gene metadata tiledb
    if not skip_gene_tiledb:
        _col_types = {}
        for col in gene_metadata.columns:
            _col_types[col] = "ascii"

        _gene_output_uri = f"{output_path}/gene_metadata"
        generate_metadata_tiledb_frame(
            _gene_output_uri, gene_metadata, column_types=_col_types
        )

        if optimize_tiledb:
            uta.optimize_tiledb_array(_gene_output_uri)

    if cell_metadata is None:
        if num_cells is None:
            warnings.warn(
                "Scanning all files to compute cell counts, this may take long",
                UserWarning,
            )
            cell_counts = uad.scan_for_cellcounts(files, num_threads=num_threads)
            num_cells = sum(cell_counts)

        cell_metadata = pd.DataFrame({"cell_index": [x for x in range(num_cells)]})

    if isinstance(cell_metadata, str):
        warnings.warn(
            "Scanning 'cell_metadata' to count number of cells, this may take long",
            UserWarning,
        )
        with open(cell_metadata) as fp:
            count = 0
            for _ in fp:
                count += 1

        num_cells = count - 1  # removing 1 for the header line
    elif isinstance(cell_metadata, pd.DataFrame):
        num_cells = len(cell_metadata)

    if num_cells is None:
        raise ValueError(
            "Cannot determine 'num_cells', we recommend setting this parameter."
        )

    # Create the cell metadata tiledb
    if not skip_cell_tiledb:
        _cell_output_uri = f"{output_path}/cell_metadata"

        if isinstance(cell_metadata, str):
            _cell_metaframe = pd.read_csv(cell_metadata, chunksize=5, header=True)
            generate_metadata_tiledb_csv(
                _cell_output_uri, cell_metadata, _cell_metaframe.columns
            )
        elif isinstance(cell_metadata, pd.DataFrame):
            _col_types = {}
            for col in gene_metadata.columns:
                _col_types[col] = "ascii"

            _to_write = gene_metadata.astype(str)

            generate_metadata_tiledb_frame(
                _cell_output_uri, _to_write, column_types=_col_types
            )

        if optimize_tiledb:
            uta.optimize_tiledb_array(_cell_output_uri)

    # create the counts metadata
    if not skip_matrix_tiledb:
        gene_idx = gene_metadata.index.tolist()
        gene_set = {}
        for i, x in enumerate(gene_idx):
            gene_set[x] = i

        _counts_uri = f"{output_path}/{layer_matrix_name}"
        uta.create_tiledb_array(
            _counts_uri,
            num_cells=num_cells,
            num_genes=num_genes,
            matrix_attr_name=layer_matrix_name,
            x_dim_dtype=cell_dim_dtype,
            y_dim_dtype=gene_dim_dtype,
            matrix_dim_dtype=matrix_dim_dtype,
        )

        offset = 0
        for fd in files:
            mat = uad.remap_anndata(
                fd,
                gene_set,
                var_gene_column=var_gene_column,
                layer_matrix_name=layer_matrix_name,
            )
            uta.write_csr_matrix_to_tiledb(
                _counts_uri, matrix=mat, row_offset=offset, value_dtype=matrix_dim_dtype
            )
            offset += int(mat.shape[0])

        if optimize_tiledb:
            uta.optimize_tiledb_array(_counts_uri)

    return CellArrDataset(dataset_path=output_path, counts_tdb_uri=layer_matrix_name)


def generate_metadata_tiledb_frame(
    output_uri: str, input: pd.DataFrame, column_types: dict = None
):
    """Generate metadata tiledb from a :pu:class:`~pandas.DataFrame`.

    Args:
        output_uri:
            TileDB URI or path to save the file.

        input:
            Input dataframe.

        column_types:
            You can specify type of each column name to cast into.
            "ascii" or str works best for most scenarios.

            Defaults to None.
    """
    _to_write = input.astype(str)
    utf.create_tiledb_frame_from_dataframe(
        output_uri, _to_write, column_types=column_types
    )


def generate_metadata_tiledb_csv(
    output_uri: str,
    input: str,
    column_dtype=str,
    chunksize=1000,
):
    """Generate a metadata tiledb from csv.

    The difference between this and ``generate_metadata_tiledb_frame``
    is when the csv is super large and it won't fit into memory.

    Args:
        output_uri:
            TileDB URI or path to save the file.

        input:
            Path to the csv file. The first row is expected to
            contain the column names.

        column_dtype:
            Dtype of the columns.
            Defaults to str.

        chunksize:
            Chunk size to read the dataframe.
            Defaults to 1000.
    """
    chunksize = 1000
    initfile = True
    offset = 0

    for chunk in pd.read_csv(input, chunksize=chunksize, header=True):
        if initfile:
            utf.create_tiledb_frame_from_column_names(
                output_uri, chunk.columns, column_dtype
            )
            initfile = False

        _to_write = chunk.astype(str)
        utf.append_to_tiledb_frame(output_uri, _to_write, offset)
        offset += len(chunk)
