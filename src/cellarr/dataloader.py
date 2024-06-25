import tiledb

__author__ = "Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

# Turn off multithreading to allow multiple pytorch dataloader workers
config = tiledb.Config()
# config["sm.compute_concurrency_level"] = 1
# config["sm.io_concurrency_level"] = 1
# config["sm.num_async_threads"] = 1
# config["sm.num_reader_threads"] = 1
# config["sm.num_tbb_threads"] = 1
# config["sm.num_writer_threads"] = 1
config["vfs.num_threads"] = 1
