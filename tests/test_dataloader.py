import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import random
from scipy.sparse import rand
import tiledb
from cellarr import (
    build_cellarrdataset,
    MatrixOptions,
)
from cellarr.dataloader import DataModule

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def generate_adata(n, d, study):
    np.random.seed(1)

    y = (100 * rand(n, d, density=0.2, format="csr")).astype(int)

    gene_index = [f"gene_{i+1}" for i in range(d)]
    var_df = pd.DataFrame({"names": gene_index}, index=gene_index)

    obs_df = pd.DataFrame({"cells": [f"cell_{j+1}" for j in range(n)]})
    obs_df["study"] = study
    labels = [f"label_{i}" for i in range(10)] + [np.nan]
    obs_df["label"] = random.choices(labels, k=n)
    obs_df["index_in_file"] = range(n)

    adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

    return adata


def test_dataloader():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, study="test1")
    adata2 = generate_adata(100, 1000, study="test2")
    adata3 = generate_adata(100, 100, study="test3")
    obs = pd.concat([adata1.obs, adata2.obs, adata3.obs])
    obs = obs.reset_index(drop=True)

    build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2, adata3],
        cell_metadata=obs,
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    datamodule = DataModule(
        dataset_path=tempdir,
        cell_metadata_uri="cell_metadata",
        gene_annotation_uri="gene_annotation",
        matrix_uri="counts",
        val_studies=["test3"],
        label_column_name="label",
        study_column_name="study",
        batch_size=100,
        lognorm=True,
        target_sum=1e4,
    )

    assert "test1" in datamodule.data_df["study"].values
    assert "test2" in datamodule.data_df["study"].values
    assert "test3" not in datamodule.data_df["study"].values
    assert "test3" in datamodule.val_df["study"].values

    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    data, labels, studies = batch
    print(data, labels, studies)
    assert len(labels) == 100
    assert len(studies) == 100
    assert data.shape == (100, len(datamodule.gene_indices))
