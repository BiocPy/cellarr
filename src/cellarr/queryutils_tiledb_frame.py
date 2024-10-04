from functools import lru_cache
from typing import List, Union
from warnings import warn

import pandas as pd
import numpy as np
from scipy import sparse as sp
import tiledb
import re

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@lru_cache
def get_schema_names_frame(tiledb_obj: tiledb.Array) -> List[str]:
    """Get Attributes from a TileDB object.

    Args:
        tiledb_obj:
            A TileDB object.

    Returns:
        List of schema attributes.
    """
    columns = []
    for i in range(tiledb_obj.schema.nattr):
        columns.append(tiledb_obj.schema.attr(i).name)

    return columns


def subset_frame(
    tiledb_obj: tiledb.Array,
    subset: Union[slice, tiledb.QueryCondition],
    columns: list,
) -> pd.DataFrame:
    """Subset a TileDB object.

    Args:
        tiledb_obj:
            TileDB object to subset.

        subset:
            A :py:class:`slice` to subset.

            Alternatively, may provide a :py:class:`~tiledb.QueryCondition`
            to subset the object.

        columns:
            List specifying the atrributes from the schema to extract.

    Returns:
        A slices `DataFrame` or a `matrix` with the subset.
    """

    if isinstance(subset, str):
        warn(
            "provided subset is string, its expected to be a 'query_condition'",
            UserWarning,
        )

        query = tiledb_obj.query(cond=subset, attrs=columns)
        data = query.df[:]
    else:
        data = tiledb_obj.df[subset][columns]

    re_null = re.compile(pattern="\x00")  # replace null strings with nan
    result = data.replace(regex=re_null, value=np.nan)
    result = result.dropna()
    return result


def _remap_index(indices: List[int]) -> List[int]:
    _map = {}
    _new_indices = []
    count = 0
    for r in list(indices):
        if r not in _map:
            _map[r] = count
            count += 1

        _new_indices.append(_map[r])

    return _new_indices, len(_map)


def subset_array(
    tiledb_obj: tiledb.Array,
    row_subset: Union[slice, list, tuple],
    column_subset: Union[slice, list, tuple],
    shape: tuple,
) -> sp.coo_matrix:
    """Subset a tiledb storing array data.

    Uses multi_index to slice.

    Args:
        tiledb_obj:
            A TileDB object

        row_subset:
            Subset along the row axis.

        column_subset:
            Subset along the column axis.

        shape:
            Shape of the entire matrix.

    Returns:
        A sparse array in a coordinate format.
    """
    data = tiledb_obj.multi_index[row_subset, column_subset]

    _cell_rows, _ = _remap_index(data["cell_index"])
    _gene_cols, _ = _remap_index(data["gene_index"])

    print("shape:", shape)
    mat = sp.coo_matrix(
        (data["data"], (_cell_rows, _gene_cols)),
        shape=shape,
    )

    return mat


def get_a_column(tiledb_obj: tiledb.Array, column_name: Union[str, List[str]]) -> list:
    """Access column(s) from the TileDB object.

    Args:
        tiledb_obj:
            A TileDB object.

        column_name:
            Name(s) of the column to access.

    Returns:
        List containing the column values.
    """
    if column_name not in get_schema_names_frame(tiledb_obj):
        raise ValueError(f"Column '{column_name}' does not exist.")

    if isinstance(column_name, str):
        column_name = [column_name]

    return tiledb_obj.query(attrs=column_name).df[:]


@lru_cache
def get_index(tiledb_obj: tiledb.Array) -> list:
    """Get the index of the TileDB object.

    Args:
        tiledb_obj:
            A TileDB object.

    Returns:
        A list containing the index values.
    """
    _index = tiledb_obj.unique_dim_values("__tiledb_rows")
    return [x.decode() for x in _index]


def _match_to_list(x: list, query: list):
    return sorted([x.index(q) for q in query])


def _is_list_strings(x: list):
    _ret = False

    if isinstance(x, (list, tuple)) and all(isinstance(y, str) for y in x):
        _ret = True

    return _ret
