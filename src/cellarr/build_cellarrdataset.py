"""Build the `CellArrDatset`.

The `CellArrDataset` method is designed to store single-cell RNA-seq
datasets but can be generalized to store any 2-dimensional experimental data.

This method creates four TileDB files in the directory specified by `output_path`:

- `gene_annotation`: A TileDB file containing feature/gene annotations.
- `sample_metadata`: A TileDB file containing sample metadata.
- `cell_metadata`: A TileDB file containing cell metadata including mapping to the samples
they are tagged with in ``sample_metadata``.
- A matrix TileDB file named by the `layer_matrix_name` parameter. This allows the package
to store multiple different matrices, e.g. 'normalized', 'scaled', 'counts' for the same
cell, gene, sample metadata attributes.

The TileDB matrix file is stored in a ``cell X gene`` orientation. This orientation
is chosen because the fastest-changing dimension as new files are added to the
collection is usually the cells rather than genes.

Process:

1. **Scan the Collection**: Scan the entire collection of files to create
a unique set of feature ids (e.g. gene symbols). Store this set as the
`gene_annotation` TileDB file.

2. **Sample Metadata**: Store sample metadata in `sample_metadata`
TileDB file. Each file is typically considered a sample, and an automatic
mapping is created between files and samples.

3. **Store Cell Metadata**: Store cell metadata in the `cell_metadata`
TileDB file.

4. **Remap and Orient Data**: For each dataset in the collection,
remap and orient the feature dimension using the feature set from Step 1.
This step ensures consistency in gene measurement and order, even if
some genes are unmeasured or ordered differently in the original experiments.

Example:

    .. code-block:: python

        import anndata
        import numpy as np
        import tempfile
        from cellarr import build_cellarrdataset, CellArrDataset, MatrixOptions

        # Create a temporary directory
        tempdir = tempfile.mkdtemp()

        # Read AnnData objects
        adata1 = anndata.read_h5ad("path/to/object1.h5ad", "r")
        # or just provide the path
        adata2 = "path/to/object2.h5ad"

        # Build CellArrDataset
        dataset = build_cellarrdataset(
            output_path=tempdir,
            files=[adata1, adata2],
            matrix_options=MatrixOptions(dtype=np.float32),
        )
"""

import os
import warnings
from typing import Dict, List, Union

import anndata
import numpy as np
import pandas as pd

from . import build_options as bopt
from . import buildutils_tiledb_array as uta
from . import buildutils_tiledb_frame as utf
from . import utils_anndata as uad
from .CellArrDataset import CellArrDataset

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


# TODO: Accept files as a dictionary with names to each dataset.
def build_cellarrdataset(
    files: List[Union[str, anndata.AnnData]],
    output_path: str,
    gene_annotation: Union[List[str], str, pd.DataFrame] = None,
    sample_metadata: Union[pd.DataFrame, str] = None,
    cell_metadata: Union[pd.DataFrame, str] = None,
    sample_metadata_options: bopt.SampleMetadataOptions = bopt.SampleMetadataOptions(),
    cell_metadata_options: bopt.CellMetadataOptions = bopt.CellMetadataOptions(),
    gene_annotation_options: bopt.GeneAnnotationOptions = bopt.GeneAnnotationOptions(),
    matrix_options: bopt.MatrixOptions = bopt.MatrixOptions(),
    optimize_tiledb: bool = True,
    num_threads: int = 1,
):
    """Generate the `CellArrDataset`.

    All files are expected to be consistent and any modifications
    to make them consistent is outside the scope of this function
    and package.

    There's a few assumptions this process makes:
    - If object in ``files`` is an :py:class:`~anndata.AnnData`
    or H5AD object, these must contain an assay matrix in the
    layers slot of the object named as ``layer_matrix_name`` parameter.
    - Feature information must contain a column defined by the parameter
    ``feature_column`` in the
    :py:class:`~cellarr.build_options.GeneAnnotationOptions.` that
    contains feature ids or gene symbols across all files.
    - If no ``cell_metadata`` is provided, we scan to count the number of cells
    and create a simple range index.
    - Each file is considered a sample and a mapping between cells and samples
    is automatically created. Hence the sample information provided must match
    the number of input files.

    Args:
        files:
            List of file paths to `H5AD` or ``AnnData`` objects.

        output_path:
            Path to where the output TileDB files should be stored.

        gene_metadata:
            A :py:class:`~pandas.DataFrame` containing the feature/gene
            annotations across all objects.

            Alternatively, may provide a path to the file containing
            a concatenated gene annotations across all datasets.
            In this case, the first row is expected to contain the
            column names and an index column containing the feature ids
            or gene symbols.

            Alternatively, a list or a dictionary of gene symbols.

            Irrespective of the input, the object will be appended
            with a ``cellarr_gene_index`` column that contains the
            gene index across all objects.

            Defaults to None, then a gene set is generated by scanning all
            objects in ``files``.

        sample_metadata:
            A :py:class:`~pandas.DataFrame` containing the sample
            metadata for each file in ``files``. Hences the number of rows
            in the dataframe must match the number of ``files``.

            Alternatively, may provide path to the file containing a
            concatenated sample metadata across all cells. In this case,
            the first row is expected to contain the column names.

            Additionally, the order of rows is expected to be in the same
            order as the input list of ``files``.

            Irrespective of the input, this object is appended with a
            ``cellarr_original_gene_list`` column that contains the original
            set of feature ids (or gene symbols) from the dataset to
            differentiate between zero-expressed vs unmeasured genes.

            Defaults to `None`, in which case, we create a simple sample
            metadata dataframe containing the list of datasets.
            Each dataset is named as ``sample_{i}`` where `i` refers to
            the index position of the object in ``files``.

        cell_metadata:
            A :py:class:`~pandas.DataFrame` containing the cell
            metadata for cells across ``files``. Hences the number of rows
            in the dataframe must match the number of cells across
            all files.

            Alternatively, may provide path to the file containing a
            concatenated cell metadata across all cells. In this case,
            the first row is expected to contain the column names.

            Additionally, the order of cells is expected to be in the same
            order as the input list of ``files``. If the input is a path,
            the file is expected to contain mappings between cells and
            datasets (or samples).

            Defaults to None, we scan all files to count the number of cells,
            then create a simple cell metadata DataFrame containing mappings from
            cells to their associated datasets. Each dataset is named as
            ``sample_{i}`` where `i` refers to the index position of
            the object in ``files``.

        sample_metadata_options:
            Optional parameters when generating ``sample_metadata`` store.

        cell_metadata_options:
            Optional parameters when generating ``cell_metadata`` store.

        gene_annotation_options:
            Optional parameters when generating ``gene_annotation`` store.

        matrix_options:
            Optional parameters when generating ``matrix`` store.

        optimize_tiledb:
            Whether to run TileDB's vaccum and consolidation (may take long).

        num_threads:
            Number of threads.
            Defaults to 1.
    """
    if not os.path.isdir(output_path):
        raise ValueError("'output_path' must be a directory.")

    _subsets = cell_metadata_options.column_types
    if _subsets is not None and len(_subsets) > 0:
        _subsets = list(_subsets.keys())

    files_cache = uad.extract_anndata_info(
        files,
        var_feature_column=gene_annotation_options.feature_column,
        obs_subset_columns=_subsets,
        num_threads=num_threads,
    )

    ####
    ## Writing gene annotation file
    ####
    if gene_annotation is None:
        warnings.warn(
            (
                "Scanning all files for feature ids (e.g. gene symbols) and cell annotations, this may take long, ",
                "Please also make sure you have enough memory.",
            ),
            UserWarning,
        )

        gene_set = uad.scan_for_features(files_cache)
        gene_set = sorted(gene_set)
        gene_annotation = pd.DataFrame({"cellarr_gene_index": gene_set}, index=gene_set)
    elif isinstance(gene_annotation, list):
        _gene_list = sorted(list(set(gene_annotation)))
        gene_annotation = pd.DataFrame(
            {"cellarr_gene_index": _gene_list}, index=_gene_list
        )
    elif isinstance(gene_annotation, str):
        gene_annotation = pd.read_csv(gene_annotation, index_col=0, header=0)
        warnings.warn(
            "Using the index of the DataFrame to collect feature ids or gene symbols...",
            UserWarning,
        )
        gene_annotation["cellarr_gene_index"] = gene_annotation.index.tolist()
    elif isinstance(gene_annotation, pd.DataFrame):
        warnings.warn(
            "Using the index of the DataFrame to collect feature ids or gene symbols...",
            UserWarning,
        )
        gene_annotation["cellarr_gene_index"] = gene_annotation.index.tolist()
    else:
        raise TypeError("'gene_annotation' is not an expected type.")

    if not isinstance(gene_annotation, pd.DataFrame):
        raise TypeError("'gene_annotation' must be a pandas dataframe.")

    if len(gene_annotation["cellarr_gene_index"].unique()) != len(
        gene_annotation["cellarr_gene_index"].tolist()
    ):
        raise ValueError(
            "'gene_annotation' must contain unique feature ids or gene symbols."
        )

    gene_annotation.reset_index(drop=True, inplace=True)

    # Create the gene annotation TileDB
    if not gene_annotation_options.skip:
        _col_types = utf.infer_column_types(
            gene_annotation, gene_annotation_options.column_types
        )

        _gene_output_uri = f"{output_path}/{gene_annotation_options.tiledb_store_name}"
        generate_metadata_tiledb_frame(
            _gene_output_uri, gene_annotation, column_types=_col_types
        )

        if optimize_tiledb:
            uta.optimize_tiledb_array(_gene_output_uri)

    ####
    ## Writing the sample metadata file
    ####
    _samples = []
    for idx, _ in enumerate(files):
        _samples.append(f"sample_{idx + 1}")

    warnings.warn(
        "Scanning all files to compute cell counts, this may take long",
        UserWarning,
    )
    cell_counts = uad.scan_for_cellcounts(files_cache)

    if sample_metadata is None:
        warnings.warn(
            "Sample metadata is not provided, each dataset in 'files' is considered a sample",
            UserWarning,
        )

        sample_metadata = pd.DataFrame({"cellarr_sample": _samples})
    elif isinstance(sample_metadata, str):
        sample_metadata = pd.read_csv(sample_metadata, header=0)
        if "cellarr_sample" not in sample_metadata.columns:
            sample_metadata["cellarr_sample"] = _samples
    elif isinstance(sample_metadata, pd.DataFrame):
        if "cellarr_sample" not in sample_metadata.columns:
            sample_metadata["cellarr_sample"] = _samples
    else:
        raise TypeError("'sample_metadata' is not an expected type.")

    if not sample_metadata_options.skip:
        warnings.warn(
            "Scanning all files for feature ids (e.g. gene symbols), this may take long",
            UserWarning,
        )
        if "cellarr_original_gene_set" not in sample_metadata.columns:
            gene_scan_set = uad.scan_for_features(files_cache, unique=False)
            gene_set_str = [",".join(x) for x in gene_scan_set]
            sample_metadata["cellarr_original_gene_set"] = gene_set_str

        if "cellarr_cell_counts" not in sample_metadata.columns:
            sample_metadata["cellarr_cell_counts"] = cell_counts

        _col_types = utf.infer_column_types(
            sample_metadata, sample_metadata_options.column_types
        )

        _sample_output_uri = (
            f"{output_path}/{sample_metadata_options.tiledb_store_name}"
        )
        generate_metadata_tiledb_frame(
            _sample_output_uri, sample_metadata, column_types=_col_types
        )

        if optimize_tiledb:
            uta.optimize_tiledb_array(_sample_output_uri)

    ####
    ## Writing the cell metadata file
    ####
    warnings.warn(
        "Scanning all files to compute cell counts, this may take long",
        UserWarning,
    )
    _sample_per_cell = []
    _cell_index_in_sample = []
    for idx, cci in enumerate(cell_counts):
        _sample_per_cell.extend([_samples[idx] for _ in range(cci)])
        _cell_index_in_sample.extend([i for i in range(cci)])

    if cell_metadata is None:
        cell_metadata = pd.DataFrame(
            {
                "cellarr_sample": _sample_per_cell,
                "cellarr_cell_index_in_sample": _cell_index_in_sample,
            }
        )

        if (
            cell_metadata_options.column_types is not None
            and len(cell_metadata_options.column_types) > 0
        ):
            pd_cell_meta = uad.scan_for_cellmetadata(files_cache)
            cell_metadata = pd.concat(
                [
                    cell_metadata.reset_index(drop=True),
                    pd_cell_meta.reset_index(drop=True),
                ],
                axis=1,
                sort=False,
            )
    elif isinstance(cell_metadata, str):
        warnings.warn(
            "Scanning 'cell_metadata' csv file to count number of cells, this may take long",
            UserWarning,
        )
        with open(cell_metadata) as fp:
            count = 0
            for _ in fp:
                count += 1

        if sum(cell_counts) != count - 1:
            raise ValueError(
                "Number of rows in the 'cell_metadata' csv does not match the number of cells across files."
            )

        warnings.warn(
            (
                "'cell_metadata' csv file is expected to contain mapping between cells and samples",
                "especially 'cellarr_sample' (which sample the cell comes from) and ",
                "'cellarr_cell_index_in_sample' (index of the cell within the sample) columns.",
            ),
            UserWarning,
        )
    elif isinstance(cell_metadata, pd.DataFrame):
        if sum(cell_counts) != len(cell_metadata):
            raise ValueError(
                "Number of rows in 'cell_metadata' does not match the number of cells across files."
            )

        if "cellarr_sample" not in cell_metadata.columns:
            cell_metadata["cellarr_sample"] = _sample_per_cell

        if "cellarr_cell_index_in_sample" not in cell_metadata.columns:
            cell_metadata["cellarr_cell_index_in_sample"] = _cell_index_in_sample

        cell_metadata.reset_index(drop=True, inplace=True)

    # Create the cell metadata TileDB
    if not cell_metadata_options.skip:
        _cell_output_uri = f"{output_path}/cell_metadata"

        if isinstance(cell_metadata, str):
            _cell_metaframe = next(pd.read_csv(cell_metadata, chunksize=5, header=0))
            _col_types = utf.infer_column_types(
                _cell_metaframe, cell_metadata_options.column_types
            )
            generate_metadata_tiledb_csv(_cell_output_uri, cell_metadata, _col_types)
        elif isinstance(cell_metadata, pd.DataFrame):
            _col_types = utf.infer_column_types(
                cell_metadata, cell_metadata_options.column_types
            )

            generate_metadata_tiledb_frame(
                _cell_output_uri, cell_metadata, column_types=_col_types
            )

        if optimize_tiledb:
            uta.optimize_tiledb_array(_cell_output_uri)

    ####
    ## Writing the matrix file
    ####
    if not matrix_options.skip:
        gene_idx = gene_annotation["cellarr_gene_index"].tolist()
        gene_set = {}
        for i, x in enumerate(gene_idx):
            gene_set[x] = i

        _counts_uri = f"{output_path}/{matrix_options.tiledb_store_name}"
        uta.create_tiledb_array(
            _counts_uri,
            matrix_attr_name=matrix_options.matrix_attr_name,
            x_dim_dtype=cell_metadata_options.dtype,
            y_dim_dtype=gene_annotation_options.dtype,
            matrix_dim_dtype=matrix_options.dtype,
        )

        offset = 0
        for fd in files:
            mat = uad.remap_anndata(
                fd,
                gene_set,
                var_feature_column=gene_annotation_options.feature_column,
                layer_matrix_name=matrix_options.matrix_name,
                consolidate_duplicate_gene_func=matrix_options.consolidate_duplicate_gene_func,
            )
            uta.write_csr_matrix_to_tiledb(
                _counts_uri,
                matrix=mat,
                row_offset=offset,
                value_dtype=matrix_options.dtype,
            )
            offset += int(mat.shape[0])

        if optimize_tiledb:
            uta.optimize_tiledb_array(_counts_uri)

    return CellArrDataset(
        dataset_path=output_path,
        sample_metadata_uri=sample_metadata_options.tiledb_store_name,
        cell_metadata_uri=cell_metadata_options.tiledb_store_name,
        gene_annotation_uri=gene_annotation_options.tiledb_store_name,
        matrix_tdb_uri=matrix_options.tiledb_store_name,
    )


def generate_metadata_tiledb_frame(
    output_uri: str, input: pd.DataFrame, column_types: dict = None
):
    """Generate metadata TileDB from a :py:class:`~pandas.DataFrame`.

    Args:
        output_uri:
            TileDB URI or path to save the file.

        input:
            Input dataframe.

        column_types:
            You can specify type of each column name to cast into.
            "ascii" or `str` works best for most scenarios.

            Defaults to None.
    """
    utf.create_tiledb_frame_from_dataframe(output_uri, input, column_types=column_types)


def generate_metadata_tiledb_csv(
    output_uri: str,
    input: str,
    column_dtype: Dict[str, np.dtype] = None,
    index_col: bool = False,
    chunksize=1000,
):
    """Generate a metadata TileDB from csv.

    The difference between this and ``generate_metadata_tiledb_frame``
    is when the csv is super large and it won't fit into memory.

    Args:
        output_uri:
            TileDB URI or path to save the file.

        input:
            Path to the csv file. The first row is expected to
            contain the column names.

        column_dtype:
            Dtype for each of the columns.
            Defaults to None.

        chunksize:
            Chunk size to read the dataframe.
            Defaults to 1000.
    """
    chunksize = 1000
    initfile = True
    offset = 0

    for chunk in pd.read_csv(input, chunksize=chunksize, header=0, index_col=index_col):
        if initfile:
            utf.create_tiledb_frame_from_chunk(
                output_uri, chunk, utf.infer_column_types(chunk, column_dtype)
            )
            initfile = False

        utf.append_to_tiledb_frame(output_uri, chunk, offset)
        offset += len(chunk)
