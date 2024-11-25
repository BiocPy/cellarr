import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledb
from cellarr import CellArrDataset, build_cellarrdataset, MatrixOptions

__author__ = "Kevin Yang"
__copyright__ = "Kevin Yang"
__license__ = "MIT"


def generate_adata(n, d, k):
    np.random.seed(1)

    z = np.random.normal(loc=np.arange(k), scale=np.arange(k) * 2, size=(n, k))
    w = np.random.normal(size=(d, k))
    y = np.dot(z, w.T)

    gene_index = [f"gene_{i+1}" for i in range(d)]
    var_df = pd.DataFrame({"names": gene_index}, index=gene_index)
    obs_df = pd.DataFrame({"cells": [f"cell1_{j+1}" for j in range(n)]})

    adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

    return adata


def test_custom_config_cellarrdataset():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, 10)
    adata2 = generate_adata(100, 1000, 100)

    dataset = build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2],
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    assert dataset is not None
    assert isinstance(dataset, CellArrDataset)

    config = tiledb.Config()
    config["sm.memory_budget"] = "100000000000"

    cd = CellArrDataset(dataset_path=tempdir, config=config)

    assert cd._matrix_tdb["counts"].schema.ctx.config()["sm.memory_budget"] == "100000000000"
    assert (
        cd._gene_annotation_tdb.schema.ctx.config()["sm.memory_budget"]
        == "100000000000"
    )
    assert (
        cd._cell_metadata_tdb.schema.ctx.config()["sm.memory_budget"] == "100000000000"
    )
    assert (
        cd._sample_metadata_tdb.schema.ctx.config()["sm.memory_budget"]
        == "100000000000"
    )
