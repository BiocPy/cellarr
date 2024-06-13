from typing import List, Union

import anndata
import numpy as np
import pandas as pd
import tiledb
from scipy.sparse import coo_matrix, csr_array, csr_matrix

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def build_tiledb_from_anndatas(
    files: List[Union[str, anndata.AnnData]], gene_set: List[str] = None
):
    pass
