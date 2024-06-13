import tiledb
from torch.utils.data import Dataset

__author__ = "Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

# Turn off multithreading to allow multiple pytorch dataloader workers
config = tiledb.Config()
config["sm.compute_concurrency_level"] = 1
config["sm.io_concurrency_level"] = 1
config["sm.num_async_threads"] = 1
config["sm.num_reader_threads"] = 1
config["sm.num_tbb_threads"] = 1
config["sm.num_writer_threads"] = 1
config["vfs.num_threads"] = 1


class CellArrDataset(Dataset):
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
