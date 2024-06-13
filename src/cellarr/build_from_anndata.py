from typing import List, Union

import anndata
import pandas as pd
import cellarr.utils_anndata as pad

import .utils_tiledb_frame as utf
import .utils_tiledb_array as uta
import .utils_anndata as uad

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def generate_tiledb(
    output_path: str,
    num_cells: int,
    num_genes: int,
    files: List[Union[str, anndata.AnnData]],
    cell_metadata: Union[pd.DataFrame, str] = None,
    gene_metadata: Union[List[str], dict, str, pd.DataFrame] = None,
    var_gene_column: str = "index",
    layer_matrix_name: str = "counts",
    skip_gene_tiledb: bool = False,
    skip_cell_tiledb: bool = False,
    skip_counts_tiledb:bool = False,
    num_threads: int = 1,
):
    if gene_metadata is None:
        gene_set = pad.scan_for_genes(
            files, var_gene_column=var_gene_column, num_threads=num_threads
        )

        gene_metadata = pd.DataFrame({"genes": gene_set}, index=gene_set)

    if isinstance(gene_metadata, list):
        _gene_list = list(set(gene_metadata))
        gene_metadata = pd.DataFrame({"genes": _gene_list}, index=_gene_list)

    if isinstance(gene_metadata, dict):
        _gene_list = list(gene_metadata.keys())
        gene_metadata = pd.DataFrame({"genes": _gene_list}, index=_gene_list)

    if isinstance(gene_metadata, str):
        gene_metadata = pd.read_csv(gene_metadata)

    if not isinstance(gene_metadata, pd.DataFrame):
        raise TypeError("'gene_metadata' must be a pandas dataframe.")
    
    if len(gene_metadata.index.unique()) != len(gene_metadata.index.tolist()):
        raise ValueError("'gene_metada' must contain a unique index")

    # Create the gene metadata tiledb
    if not skip_gene_tiledb:
        _col_types = {}
        for col in gene_metadata.columns:
            _col_types[col] = "ascii"
        

        generate_metadata_tiledb_frame(
            f"{output_path}/gene_metadata", _to_write, column_types=_col_types
        )

    # Create the cell metadata tiledb
    if not skip_cell_tiledb:
        _cell_output_uri = f"{output_path}/cell_metadata"
        if isinstance(cell_metadata, str):
            _cell_metaframe = pd.read_csv(cell_metadata, chunksize=5)
            generate_metadata_tiledb_csv(_cell_output_uri, cell_metadata, _cell_metaframe.columns)
        elif isinstance(cell_metadata, pd.DataFrame):
            _col_types = {}
            for col in gene_metadata.columns:
                _col_types[col] = "ascii"
            
            _to_write = gene_metadata.astype(str)

            generate_metadata_tiledb_frame(
                f"{output_path}/gene_metadata", _to_write, column_types=_col_types
            )

    # create the counts metadata
    if not skip_counts_tiledb:
        gene_idx = gene_metadata.index.tolist()
        gene_set = {}
        for i, x in enumerate(gene_idx):
            gene_set[x] = i

        _counts_uri = f"{output_path}/{layer_matrix_name}"
        uta.create_tiledb_array(_counts_uri, num_cells=num_cells, num_genes=num_genes, matrix_attr_name=layer_matrix_name)
        offset = 0

        for fd in files:
            mat = uad.remap_anndata(fd, gene_set, var_gene_column=var_gene_column, layer_matrix_name=layer_matrix_name)
            uta.write_csr_matrix_to_tiledb(_counts_uri, matrix=mat, row_offset=offset)
            offset += int(mat.shape[0])

def generate_metadata_tiledb_frame(
    output_uri: str, input: pd.DataFrame, column_types: dict = None
):
    _to_write = input.astype(str)
    utf.create_tiledb_frame_from_dataframe(output_uri, _to_write, column_types=column_types)


def generate_metadata_tiledb_csv(
    output_uri: str, input: str, column_names: List[str], column_dtype = str, chunksize = 1000

):
    chunksize = 1000
    initfile = True
    offset = 0

    for chunk in pd.read_csv(input, chunksize=chunksize):
        if initfile:
            utf.create_tiledb_frame_from_column_names(output_uri, column_names, column_dtype)
            initfile = False

        _to_write = chunk.astype(str)
        utf.append_to_tiledb_frame(output_uri, _to_write, offset)
        offset += len(chunk)
