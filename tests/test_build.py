import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledb
from cellarr import build_cellarrdataset, MatrixOptions

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


def test_build_cellarrdataset():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, 10)
    adata2 = generate_adata(100, 1000, 100)

    build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2],
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    cfp = tiledb.open(f"{tempdir}/counts", "r")
    gfp = tiledb.open(f"{tempdir}/gene_annotation", "r")

    genes = gfp.df[:]

    assert len(genes) == 1000

    gene_list = ["gene_1", "gene_95", "gene_50"]
    _genes_from_tile = genes["cellarr_gene_index"].tolist()
    gene_indices_tdb = sorted([_genes_from_tile.index(x) for x in gene_list])

    adata1_gene_indices = sorted(
        [adata1.var.index.tolist().index(x) for x in gene_list]
    )

    adata2_gene_indices = sorted(
        [adata2.var.index.tolist().index(x) for x in gene_list]
    )

    assert np.allclose(
        cfp.multi_index[0, gene_indices_tdb]["data"],
        adata1.layers["counts"][0, adata1_gene_indices],
    )
    assert np.allclose(
        cfp.multi_index[1000, gene_indices_tdb]["data"],
        adata2.layers["counts"][0, adata2_gene_indices],
    )

    sfp = tiledb.open(f"{tempdir}/sample_metadata", "r")
    samples = sfp.df[:]
    assert len(samples) == 2

    cellfp = tiledb.open(f"{tempdir}/cell_metadata", "r")
    cell_df = cellfp.df[:]
    assert len(cell_df) == 1100
