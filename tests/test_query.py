import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledb
from cellarr import CellArrDataset, build_cellarrdataset, MatrixOptions

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
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


def test_query_cellarrdataset():
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

    cd = CellArrDataset(dataset_path=tempdir)

    gene_list = ["gene_1", "gene_95", "gene_50"]

    gfp = tiledb.open(f"{tempdir}/gene_annotation", "r")

    genes = gfp.df[:]
    _genes_from_tile = genes["cellarr_gene_index"].tolist()
    gene_indices_tdb = sorted([_genes_from_tile.index(x) for x in gene_list])

    adata1_gene_indices = sorted(
        [adata1.var.index.tolist().index(x) for x in gene_list]
    )

    adata2_gene_indices = sorted(
        [adata2.var.index.tolist().index(x) for x in gene_list]
    )

    result1 = cd[0, gene_list]

    assert result1 is not None

    assert np.allclose(
        result1.matrix.data,
        adata1.layers["counts"][0, adata1_gene_indices],
    )

    result2 = cd[1000, gene_list]

    assert np.allclose(
        result2.matrix.data,
        adata2.layers["counts"][0, adata2_gene_indices],
    )
