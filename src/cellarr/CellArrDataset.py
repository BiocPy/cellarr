import os
from typing import List, Union, Sequence

import pandas as pd
import tiledb

from . import queryutils_tiledb_frame as qtd

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrDataset:
    """A class that represent a collection of cells and their associated metadata 
    in a TileDB backed store.
    """

    def __init__(
        self,
        dataset_path: str,
        matrix_tdb_uri: str = "counts",
        gene_annotation_uri: str = "gene_annotation",
        cell_metadata_uri: str = "cell_metadata",
    ):
        """Initialize a ``CellArrDataset``.

        Args:
            dataset_path:
                Path to the directory containing the tiledb stores.
                Usually the ``output_path`` from the
                :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

            counts_tdb_uri:
                Relative path to matrix store.

            gene_annotation_uri:
                Relative path to gene annotation store.

            cell_metadata_uri:
                Relative path to cell metadata store.
        """

        if not os.path.isdir(dataset_path):
            raise ValueError("'dataset_path' is not a directory.")

        self._dataset_path = dataset_path
        # TODO: Maybe switch to on-demand loading of these objects
        self._matrix_tdb_tdb = tiledb.open(f"{dataset_path}/{matrix_tdb_uri}", "r")
        self._gene_annotation_tdb = tiledb.open(
            f"{dataset_path}/{gene_annotation_uri}", "r"
        )
        self._cell_metadata_tdb = tiledb.open(
            f"{dataset_path}/{cell_metadata_uri}", "r"
        )

    def __del__(self):
        self._matrix_tdb_tdb.close()
        self._gene_annotation_tdb.close()
        self._cell_metadata_tdb.close()

    def get_cell_metadata_columns(self) -> List[str]:
        """Get column names from ``cell_metadata`` store.

        Returns:
            List of available metadata columns.
        """
        return qtd.get_schema_names_frame(self._cell_metadata_tdb)

    def get_cell_metadata_column(self, column_name: str) -> list:
        """Access a column from the ``cell_metadata`` store.

        Args:
            column_name:
                Name of the column or attribute. Usually one of the column names
                from of :py:meth:`~get_cell_metadata_columns`.

        Returns:
            A list of values for this column.
        """
        return qtd.get_a_column(self._cell_metadata_tdb, column_name=column_name)

    def get_cell_subset(
        self, subset: Union[slice, tiledb.QueryCondition], columns=None
    ) -> pd.DataFrame:
        """Slice the ``cell_metadata`` store.

        Args:
            subset:
                A list of integer indices to subset the ``cell_metadata``
                store.

                Alternatively, may also provide a
                :py:class:`tiledb.QueryCondition` to query the store.

            columns:
                List of specific column names to access.

                Defaults to None, in which case all columns are extracted.

        Returns:
            A pandas Dataframe of the subset.
        """
        return qtd.subset_frame(self._cell_metadata_tdb, subset=subset, columns=columns)

    def get_gene_metadata_columns(self) -> List[str]:
        """Get annotation column names from ``gene_metadata`` store.

        Returns:
            List of available annotations.
        """
        return qtd.get_schema_names_frame(self._gene_annotation_tdb)

    def get_gene_metadata_column(self, column_name: str):
        """Access a column from the ``gene_metadata`` store.

        Args:
            column_name:
                Name of the column or attribute. Usually one of the column names
                from of :py:meth:`~get_gene_metadata_columns`.

        Returns:
            A list of values for this column.
        """
        return qtd.get_a_column(self._gene_annotation_tdb, column_name=column_name)

    def get_gene_metadata_index(self):
        """Get index of the ``gene_metadata`` store. This typically should store all unique gene symbols.

        Returns:
            List of unique symbols.
        """
        return qtd.get_index(self._gene_annotation_tdb)

    def _get_indices_for_gene_list(self, query: list) -> List[int]:
        _gene_index = self.get_gene_metadata_index()
        return qtd._match_to_list(_gene_index, query=query)

    def get_gene_subset(
        self, subset: Union[slice, List[str], tiledb.QueryCondition], columns=None
    ):
        """Slice the ``gene_metadata`` store.

        Args:
            subset:
                A list of integer indices to subset the ``gene_metadata``
                store.

                Alternatively, may provide a
                :py:class:`tiledb.QueryCondition` to query the store.

                Alternatively, may provide a list of strings to match with
                the index of ``gene_metadata`` store.

            columns:
                List of specific column names to access.

                Defaults to None, in which case all columns are extracted.

        Returns:
            A pandas Dataframe of the subset.
        """

        if qtd._is_list_strings(subset):
            subset = self._get_indices_for_gene_list(subset)

        return qtd.subset_frame(self._gene_annotation_tdb, subset=subset, columns=columns)

    def get_slice(
        self,
        cell_subset: Union[slice, tiledb.QueryCondition],
        gene_subset: Union[slice, List[str], tiledb.QueryCondition],
    ):
        _csubset = self.get_cell_subset(cell_subset)
        _cell_indices = _csubset.index.tolist()

        _gsubset = self.get_gene_subset(gene_subset)
        _gene_indices = _gsubset.index.tolist()

        return self._matrix_tdb_tdb.multi_index[_cell_indices, _gene_indices]

    def __getitem__(
        self,
        args: Union[int, str, Sequence, tuple],
    ):
        """Subset a ``CellArrDataset``.

        Args:
            args:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted.

                Alternatively a tuple of length 1. The first entry specifies
                the rows to retain based on their names or indices.

                Alternatively a tuple of length 2. The first entry specifies
                the rows to retain, while the second entry specifies the
                columns to retain, based on their names or indices.

        Raises:
            ValueError:
                If too many or too few slices provided.
        """
        if isinstance(args, (str, int)):
            return self.get_slice(args, slice(None))

        if isinstance(args, tuple):
            if len(args) == 0:
                raise ValueError("At least one slicing argument must be provided.")

            if len(args) == 1:
                return self.get_slice(args[0], slice(None))
            elif len(args) == 2:
                return self.get_slice(args[0], args[1])
            else:
                raise ValueError(
                    f"`{type(self).__name__}` only supports 2-dimensional slicing."
                )

        raise TypeError(
            "args must be a sequence or a scalar integer or string or a tuple of atmost 2 values."
        )
