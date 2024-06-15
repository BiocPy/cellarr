import os
import shutil
from typing import List

import pandas as pd
import tiledb

__author__ = "Jayaram Kancherla, Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_tiledb_frame_from_column_names(
    tiledb_uri_path: str, column_names: List[str], column_dtype=str
):
    """Create a tiledb file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata tiledb file.

        column_names:
            Column names of the data frame.

        column_dtype:
            Type for the columns, usually str.
            Defaults to string.
    """
    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    df = pd.DataFrame(columns=list(column_names), dtype=column_dtype)
    for c in df.columns:
        df.loc[0, c] = "None"

    tiledb.from_pandas(tiledb_uri_path, df, mode="schema_only", full_domain=True)


def create_tiledb_frame_from_dataframe(
    tiledb_uri_path: str, frame: List[str], column_types=dict
):
    """Create a tiledb file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata tiledb file.

        column_names:
            Column names of the data frame.

        column_types:
            Dictionary specifying the column types for each
            column in the frame.
    """
    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    tiledb.from_pandas(tiledb_uri_path, dataframe=frame, column_types=column_types)


def append_to_tiledb_frame(
    tiledb_uri_path: str, frame: pd.DataFrame, row_offset: int = 0
):
    """Create a tiledb file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata tiledb file.

        frame:
            Pandas Dataframe to append to tiledb.

        row_offset:
            Row offset to append new rows to.
            Defaults to 0.
    """
    tiledb.from_pandas(
        tiledb_uri_path, dataframe=frame, mode="append", row_start_idx=row_offset
    )
