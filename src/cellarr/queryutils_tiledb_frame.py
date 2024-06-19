from functools import lru_cache
from typing import List, Union
from warnings import warn

import pandas as pd
import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@lru_cache
def get_schema_names_frame(tiledb_obj: tiledb.Array) -> List[str]:
    columns = []
    for i in range(tiledb_obj.schema.nattr):
        columns.append(tiledb_obj.schema.attr(i).name)

    return columns


def subset_frame(
    tiledb_obj: tiledb.Array,
    subset: Union[slice, tiledb.QueryCondition],
    columns: Union[str, list] = None,
) -> pd.DataFrame:

    _avail_columns = get_schema_names_frame(tiledb_obj)

    if columns is None:
        columns = _avail_columns
    else:
        _not_avail = []
        for col in columns:
            if col not in _avail_columns:
                _not_avail.append(col)

        if len(_not_avail) > 0:
            raise ValueError(f"Columns '{', '.join(_not_avail)}' are not available.")

    if isinstance(columns, str):
        warn(
            "provided subset is string, its expected to be a 'query_condition'",
            UserWarning,
        )

        query = tiledb_obj.query(cond=subset, attrs=columns)
        data = query.df[:]
    else:
        data = query.df[subset, columns]
    result = data.dropna()
    return result


def get_a_column(tiledb_obj: tiledb.Array, column_name: str) -> list:
    if column_name not in get_schema_names_frame(tiledb_obj):
        raise ValueError(f"Column '{column_name}' does not exist.")

    return tiledb_obj.query(attrs=[column_name]).df[:]


@lru_cache
def get_index(tiledb_obj: tiledb.Array) -> list:
    _index = tiledb_obj.unique_dim_values("__tiledb_rows")
    return [x.decode() for x in _index]


def _match_to_list(x: list, query: list):
    return sorted([x.index(x) for x in query])


def _is_list_strings(x: list):
    _ret = False

    if isinstance(x, (list, tuple)) and all(isinstance(y, str) for y in x):
        _ret = True

    return _ret
