import tempfile

import anndata
import numpy as np
import pandas as pd
import cellarr.utils_anndata
import pytest
import tiledb
import cellarr

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_consolidate_symbols():
    np.random.seed(1)

    n = 100
    y = np.eye(n, dtype=int)
    gene_index = [f"gene_{(i % 10)+1}" for i in range(n)]
    cmat, groups = cellarr.utils_anndata.consolidate_duplicate_symbols(
        y, gene_index, consolidate_duplicate_gene_func=sum
    )

    assert len(groups) == 10
    assert cmat.shape[1] == 10
    assert cmat[:, 1].sum() == 10
