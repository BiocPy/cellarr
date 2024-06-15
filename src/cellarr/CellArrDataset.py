import os
from typing import Union

import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrDataset:
    """A class that represent a collection of cells in TileDB."""

    def __init__(
        self,
        dataset_path: str,
        counts_tdb_uri: str = "counts",
        gene_metadata_uri: str = "gene_metadata",
        cell_metadata_uri: str = "cell_metadata",
    ):
        """Initialize a ``CellArr`` dataset.

        Args:
            counts_tdb_uri:
                Path to counts TileDB.

            gene_metadata_uri:
                Path to gene metadata TileDB.

            cell_metadata_uri:
                Path to cell metadata TileDB.
        """

        if not os.path.isdir(dataset_path):
            raise ValueError("'dataset_path' is not a directory.")

        self._dataset_path = dataset_path
        # TODO: Maybe switch to on-demand loading of these objects
        self._counts_tdb_tdb = tiledb.open(f"{dataset_path}/{counts_tdb_uri}", "r")
        self._gene_metadata_tdb = tiledb.open(
            f"{dataset_path}/{gene_metadata_uri}", "r"
        )
        self._cell_metadata_tdb = tiledb.open(
            f"{dataset_path}/{cell_metadata_uri}", "r"
        )

    def __del__(self):
        self._counts_tdb_tdb.close()
        self._gene_metadata_tdb.close()
        self._cell_metadata_tdb.close()

    # TODO:
    # Methods to implement
    # search by gene
    # search by cell metadata
    # slice counts after search

    def get_cell_metadata_columns(self):
        columns = []
        for i in range(self._cell_metadata_tdb.schema.nattr):
            columns.append(self._cell_metadata_tdb.schema.attr(i).name)

        return columns

    def get_cell_metadata_column(self, column_name: str):
        return self._cell_metadata_tdb.query(attrs=[column_name]).df[:]

    def get_cell_subset(
        self, subset: Union[slice, tiledb.QueryCondition], columns=None
    ):
        if columns is None:
            columns = self.get_cell_metadata_columns()

        query = self._cell_metadata_tdb.query(cond=subset, attrs=columns)
        data = query.df[:]
        result = data.dropna()
        return result

    def get_gene_metadata_columns(self):
        columns = []
        for i in range(self._gene_metadata_tdb.schema.nattr):
            columns.append(self._gene_metadata_tdb.schema.attr(i).name)

        return columns

    def get_gene_metadata_column(self, column_name: str):
        return self._gene_metadata_tdb.query(attrs=[column_name]).df[:]

    def get_gene_subset(
        self, subset: Union[slice, tiledb.QueryCondition], columns=None
    ):
        if columns is None:
            columns = self.get_gene_metadata_columns()

        query = self._gene_metadata_tdb.query(cond=subset, attrs=columns)
        data = query.df[:]
        result = data.dropna()
        return result

    def get_slice(
        self,
        cell_subset: Union[slice, tiledb.QueryCondition],
        gene_subset: Union[slice, tiledb.QueryCondition],
    ):
        _csubset = self.get_cell_subset(cell_subset)
        _cell_indices = _csubset.index.tolist()

        _gsubset = self.get_gene_subset(gene_subset)
        _gene_indices = _gsubset.index.tolist()

        return self._counts_tdb_tdb.multi_index[_cell_indices, _gene_indices]
