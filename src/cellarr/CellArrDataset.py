import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

class CellArrDataset():
    """A class that represent a collection of cells in TileDB."""

    def __init__(
        self,
        counts_tdb_uri: tiledb.SparseArray,
        gene_metadata_uri: tiledb.Array,
        cell_metadata_uri: tiledb.Array,
    ):
        """Initialize a CellArr dataset.

        Args:
            counts_tdb_uri:
                Counts TileDB.

            gene_metadata_uri:
                Gene Metadata TileDB.

            cell_metadata_uri:
                cell Metadata TileDB.
        """

    # TODO: 
    # Methods to implement
    # search by gene
    # search by cell metadata
    # slice counts after search