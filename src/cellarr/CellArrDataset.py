"""Query the `CellArrDataset`.

This class provides methods to access the directory containing the
generated TileDB files usually using the
:py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

Example:

    .. code-block:: python

        from cellarr import CellArrDataset

        cd = CellArrDataset(dataset_path="/path/to/cellar/dir")
        gene_list = ["gene_1", "gene_95", "gene_50"]
        result1 = cd[0, gene_list]

        print(result1)
"""

import os
from typing import List, Sequence, Union

import pandas as pd
import tiledb

from . import queryutils_tiledb_frame as qtd
from .CellArrDatasetSlice import CellArrDatasetSlice

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrDataset:
    """A class that represent a collection of cells and their associated metadata in a TileDB backed store."""

    def __init__(
        self,
        dataset_path: str,
        matrix_tdb_uri: str = "counts",
        gene_annotation_uri: str = "gene_annotation",
        cell_metadata_uri: str = "cell_metadata",
        sample_metadata_uri: str = "sample_metadata",
    ):
        """Initialize a ``CellArrDataset``.

        Args:
            dataset_path:
                Path to the directory containing the TileDB stores.
                Usually the ``output_path`` from the
                :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

            counts_tdb_uri:
                Relative path to matrix store.

            gene_annotation_uri:
                Relative path to gene annotation store.

            cell_metadata_uri:
                Relative path to cell metadata store.

            sample_metadata_uri:
                Relative path to sample metadata store.
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
        self._sample_metadata_tdb = tiledb.open(
            f"{dataset_path}/{sample_metadata_uri}", "r"
        )

    def __del__(self):
        self._matrix_tdb_tdb.close()
        self._gene_annotation_tdb.close()
        self._cell_metadata_tdb.close()
        self._sample_metadata_tdb.close()

    ####
    ## Subset methods for the `cell_metadata` TileDB file.
    ####
    def get_cell_metadata_columns(self) -> List[str]:
        """Get column names from ``cell_metadata`` store.

        Returns:
            List of available metadata columns.
        """
        return qtd.get_schema_names_frame(self._cell_metadata_tdb)

    def get_cell_metadata_column(self, column_name: str) -> pd.DataFrame:
        """Access a column from the ``cell_metadata`` store.

        Args:
            column_name:
                Name of the column or attribute. Usually one of the column names
                from of :py:meth:`~get_cell_metadata_columns`.

        Returns:
            A list of values for this column.
        """
        res = qtd.get_a_column(self._cell_metadata_tdb, column_name=column_name)
        return res[column_name]

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

        if isinstance(columns, str):
            columns = [columns]

        if columns is None:
            columns = self.get_cell_metadata_columns()
        else:
            _not_avail = []
            for col in columns:
                if col not in self.get_cell_metadata_columns():
                    _not_avail.append(col)

            if len(_not_avail) > 0:
                raise ValueError(
                    f"Columns '{', '.join(_not_avail)}' are not available."
                )

        return qtd.subset_frame(self._cell_metadata_tdb, subset=subset, columns=columns)

    ####
    ## Subset methods for the `gene_annotation` TileDB file.
    ####
    def get_gene_annotation_columns(self) -> List[str]:
        """Get annotation column names from ``gene_annotation`` store.

        Returns:
            List of available annotations.
        """
        return qtd.get_schema_names_frame(self._gene_annotation_tdb)

    def get_gene_annotation_column(self, column_name: str) -> pd.DataFrame:
        """Access a column from the ``gene_annotation`` store.

        Args:
            column_name:
                Name of the column or attribute. Usually one of the column names
                from of :py:meth:`~get_gene_annotation_columns`.

        Returns:
            A list of values for this column.
        """
        res = qtd.get_a_column(self._gene_annotation_tdb, column_name=column_name)
        return res[column_name]

    def get_gene_annotation_index(self) -> List[str]:
        """Get index of the ``gene_annotation`` store.

        Returns:
            List of unique symbols.
        """
        res = qtd.get_a_column(self._gene_annotation_tdb, "cellarr_gene_index")
        return res["cellarr_gene_index"].tolist()

    def _get_indices_for_gene_list(self, query: list) -> List[int]:
        _gene_index = self.get_gene_annotation_index()
        return qtd._match_to_list(_gene_index, query=query)

    def get_gene_subset(
        self, subset: Union[slice, List[str], tiledb.QueryCondition], columns=None
    ) -> pd.DataFrame:
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

        if isinstance(columns, str):
            columns = [columns]

        if columns is None:
            columns = self.get_gene_annotation_columns()
        else:
            _not_avail = []
            for col in columns:
                if col not in self.get_gene_annotation_columns():
                    _not_avail.append(col)

            if len(_not_avail) > 0:
                raise ValueError(
                    f"Columns '{', '.join(_not_avail)}' are not available."
                )

        if qtd._is_list_strings(subset):
            subset = self._get_indices_for_gene_list(subset)

        return qtd.subset_frame(
            self._gene_annotation_tdb, subset=subset, columns=columns
        )

    ####
    ## Subset methods for the `sample_metadata` TileDB file.
    ####
    def get_sample_metadata_columns(self) -> List[str]:
        """Get column names from ``sample_metadata`` store.

        Returns:
            List of available metadata columns.
        """
        return qtd.get_schema_names_frame(self._sample_metadata_tdb)

    def get_sample_metadata_column(self, column_name: str) -> pd.DataFrame:
        """Access a column from the ``sample_metadata`` store.

        Args:
            column_name:
                Name of the column or attribute. Usually one of the column names
                from of :py:meth:`~get_sample_metadata_columns`.

        Returns:
            A list of values for this column.
        """
        res = qtd.get_a_column(self._sample_metadata_tdb, column_name=column_name)
        return res[column_name]

    def get_sample_subset(
        self, subset: Union[slice, tiledb.QueryCondition], columns=None
    ) -> pd.DataFrame:
        """Slice the ``sample_metadata`` store.

        Args:
            subset:
                A list of integer indices to subset the ``sample_metadata``
                store.

                Alternatively, may also provide a
                :py:class:`tiledb.QueryCondition` to query the store.

            columns:
                List of specific column names to access.

                Defaults to None, in which case all columns are extracted.

        Returns:
            A pandas Dataframe of the subset.
        """
        if isinstance(columns, str):
            columns = [columns]

        if columns is None:
            columns = self.get_sample_metadata_columns()
        else:
            _not_avail = []
            for col in columns:
                if col not in self.get_sample_metadata_columns():
                    _not_avail.append(col)

            if len(_not_avail) > 0:
                raise ValueError(
                    f"Columns '{', '.join(_not_avail)}' are not available."
                )

        return qtd.subset_frame(
            self._sample_metadata_tdb, subset=subset, columns=columns
        )

    ####
    ## Subset methods for the `matrix` TileDB file.
    ####
    def get_matrix_subset(self, subset: Union[int, Sequence, tuple]) -> pd.DataFrame:
        """Slice the ``sample_metadata`` store.

        Args:
            subset:
                Any `slice`supported by TileDB's array slicing.
                For more info refer to
                <TileDB docs https://docs.tiledb.com/main/how-to/arrays/reading-arrays/basic-reading>_.

        Returns:
            A pandas Dataframe of the subset.
        """
        if isinstance(subset, (str, int)):
            return qtd.subset_array(
                self._matrix_tdb_tdb,
                subset,
                slice(None),
                shape=(len(subset), self.shape[1]),
            )

        if isinstance(subset, tuple):
            if len(subset) == 0:
                raise ValueError("At least one slicing argument must be provided.")

            if len(subset) == 1:
                return qtd.subset_array(
                    self._matrix_tdb_tdb,
                    subset[0],
                    slice(None),
                    shape=(len(subset[0]), self.shape[1]),
                )
            elif len(subset) == 2:
                return qtd.subset_array(
                    self._matrix_tdb_tdb,
                    subset[0],
                    subset[1],
                    shape=(len(subset[0]), len(subset[1])),
                )
            else:
                raise ValueError(
                    f"`{type(self).__name__}` only supports 2-dimensional slicing."
                )

    ####
    ## Subset methods by cell and gene dimensions.
    ####
    def get_slice(
        self,
        cell_subset: Union[slice, tiledb.QueryCondition],
        gene_subset: Union[slice, List[str], tiledb.QueryCondition],
    ) -> CellArrDatasetSlice:
        """Subset a ``CellArrDataset``.

        Args:
            cell_subset:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the rows (or cells) to retain.

            cell_subset:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the columns (or features/genes) to retain.

        Returns:
            A :py:class:`~cellarr.CellArrDatasetSlice.CellArrDatasetSlice` object
            containing the `cell_metadata`, `gene_annotation` and the matrix for
            the given slice ranges.
        """
        _csubset = self.get_cell_subset(cell_subset)
        _cell_indices = _csubset.index.tolist()

        _gsubset = self.get_gene_subset(gene_subset)
        _gene_indices = _gsubset.index.tolist()

        _msubset = self.get_matrix_subset((_cell_indices, _gene_indices))

        return CellArrDatasetSlice(
            _csubset,
            _gsubset,
            _msubset,
        )

    ####
    ## Dunder method to use `[]` operator.
    ####
    def __getitem__(
        self,
        args: Union[int, Sequence, tuple],
    ) -> CellArrDatasetSlice:
        """Subset a ``CellArrDataset``.

        Mostly an alias to :py:meth:`~.get_slice`.

        Args:
            args:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted.

                Alternatively a tuple of length 1. The first entry specifies
                the rows (or cells) to retain based on their names or indices.

                Alternatively a tuple of length 2. The first entry specifies
                the rows (or cells) to retain, while the second entry specifies the
                columns (or features/genes) to retain, based on their names or indices.

        Raises:
            ValueError:
                If too many or too few slices provided.

        Returns:
            A :py:class:`~cellarr.CellArrDatasetSlice.CellArrDatasetSlice` object
            containing the `cell_metadata`, `gene_annotation` and the matrix.
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

    ####
    ## Misc methods.
    ####
    @property
    def shape(self):
        return (
            self._cell_metadata_tdb.nonempty_domain()[0][1] + 1,
            self._gene_annotation_tdb.nonempty_domain()[0][1] + 1,
        )

    def __len__(self):
        return self.shape[0]

    ####
    ## Printing.
    ####

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}(number_of_rows={self.shape[0]}"
        output += f", number_of_columns={self.shape[1]}"
        output += ", at path=" + self._dataset_path

        output += ")"
        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        output += f"number_of_rows: {self.shape[0]}\n"
        output += f"number_of_columns: {self.shape[1]}\n"
        output += f"path: '{self._dataset_path}'\n"

        return output
