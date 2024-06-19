# import tempfile

# import anndata
# import numpy as np
# import pandas as pd
# import pytest
# import tiledb
# from cellarr import CellArrDataset, build_cellarrdataset

# __author__ = "Jayaram Kancherla"
# __copyright__ = "Jayaram Kancherla"
# __license__ = "MIT"


# def generate_adata(n, d, k):
#     np.random.seed(1)

#     z = np.random.normal(loc=np.arange(k), scale=np.arange(k) * 2, size=(n, k))
#     w = np.random.normal(size=(d, k))
#     y = np.dot(z, w.T)

#     gene_index = [f"gene_{i+1}" for i in range(d)]
#     var_df = pd.DataFrame({"names": gene_index}, index=gene_index)
#     obs_df = pd.DataFrame({"cells": [f"cell1_{j+1}" for j in range(n)]})

#     adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

#     return adata


# def test_query_cellarrdataset():
#     tempdir = tempfile.mkdtemp()

#     adata1 = generate_adata(1000, 100, 10)
#     adata2 = generate_adata(100, 1000, 100)

#     dataset = build_cellarrdataset(
#         output_path=tempdir, files=[adata1, adata2], matrix_dim_dtype=np.float32
#     )

#     assert dataset is not None
#     assert isinstance(dataset, CellArrDataset)

#     cfp = tiledb.open(f"{tempdir}/counts", "r")
#     gfp = tiledb.open(f"{tempdir}/gene_metadata", "r")

#     cd = CellArrDataset(dataset_path=tempdir)

#     gene_list = ["gene_1", "gene_95", "gene_50"]
#     result1 = cd[0, gene_list]

#     assert np.allclose(
#         cfp.multi_index[0, gene_indices_tdb]["counts"],
#         adata1.layers["counts"][0, adata1_gene_indices],
#     )
#     assert np.allclose(
#         cfp.multi_index[1000, gene_indices_tdb]["counts"],
#         adata2.layers["counts"][0, adata2_gene_indices],
#     )
