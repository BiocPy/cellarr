---
file_format: mystnb
kernelspec:
  name: python
---

# Cell Arrays

Cell Arrays is a Python package that provides a TileDB-backed store for large collections of
genomic experimental data, such as millions of cells across multiple single-cell experiment objects.

# Install the package

To get started, install the package from [PyPI](https://pypi.org/project/cellarr/)

```bash
pip install cellarr
```

# Build the `CellArrDataset`

The `CellArrDataset` method is designed to store single-cell RNA-seq
datasets but can be generalized to store any 2-dimensional experimental data.

This method creates four TileDB files in the directory specified by `output_path`:

- `gene_annotation`: A TileDB file containing feature/gene annotations.
- `sample_metadata`: A TileDB file containing sample metadata.
- `cell_metadata`: A TileDB file containing cell metadata including mapping to the samples
they are tagged with in ``sample_metadata``.
- A matrix TileDB file named by the `layer_matrix_name` parameter. This allows the package
to store multiple different matrices, e.g. normalized, scaled for the same cell, gene, sample
metadata attributes.

The TileDB matrix file is stored in a ``cell X gene`` orientation. This orientation
is chosen because the fastest-changing dimension as new files are added to the
collection is usually the cells rather than genes.

Process:

1. **Scan the Collection**: Scan the entire collection of files to create
a unique set of feature ids (e.g. gene symbols). Store this set as the
`gene_annotation` TileDB file.

2. **Sample Metadata**: Store sample metadata in `sample_metadata`
TileDB file. Each file is typically considered a sample, and an automatic
mapping is created between files and samples.

3. **Store Cell Metadata**: Store cell metadata in the `cell_metadata`
TileDB file.

4. **Remap and Orient Data**: For each dataset in the collection,
remap and orient the feature dimension using the feature set from Step 1.
This step ensures consistency in gene measurement and order, even if
some genes are unmeasured or ordered differently in the original experiments.

:::{note}
Check out the [reference](https://biocpy.github.io/cellarr/api/cellarr#module-cellarr.build_cellarrdataset) documentation for modifying the parameters for any of these steps.
:::

First lets mock a few `AnnData` objects:

```{code-cell}
import anndata
import numpy as np
import pandas as pd

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

adata1 = generate_adata(1000, 100, 10)
adata2 = generate_adata(100, 1000, 100)

print("datasets")
print(adata1, adata2)
```

To build a `CellArrDataset` from a collection of `H5AD` or `AnnData` objects:

```{code-cell}
import anndata
import numpy as np
import tempfile
from cellarr import build_cellarrdataset, CellArrDataset, MatrixOptions

# Create a temporary directory
tempdir = tempfile.mkdtemp()

# # Read AnnData objects
# adata1 = anndata.read_h5ad("path/to/object1.h5ad", "r")
# # or just provide the path
# adata2 = "path/to/object2.h5ad"

# Build CellArrDataset
dataset = build_cellarrdataset(
    output_path=tempdir,
    files=[adata1, adata2],
    matrix_options=MatrixOptions(dtype=np.float32),
)

print(dataset)
```

:::{important}
All files are expected to be consistent and any modifications
to make them consistent is outside the scope of this function
and package.

There's a few assumptions this process makes:

- If object in ``files`` is an `AnnData`
or H5AD object, these must contain an assay matrix in the
layers slot of the object named as ``layer_matrix_name`` parameter.

- Feature information must contain a column defined by the parameter
``feature_column`` in the
`GeneAnnotationOptions.` that
contains feature ids or gene symbols across all files.

- If no ``cell_metadata`` is provided, we scan to count the number of cells
and create a simple range index.

- Each file is considered a sample and a mapping between cells and samples
is automatically created. Hence the sample information provided must match
the number of input files.
:::

# Query a `CellArrDataset`

Users have the option to reuse the `dataset` object retuned when building the dataset or by creating a `CellArrDataset` object by initializing it to the path where the files were created.

```{code-cell}
# Create a CellArrDataset object from the existing dataset
dataset = CellArrDataset(dataset_path=tempdir)

# Query data from the dataset
expression_data = dataset[10, ["gene_1", "gene_10", "gene_500"]]

print("matrix slice:")
print(expression_data.matrix)

print("\n\n gene_annotation slice:")
print(expression_data.gene_annotation)

print("\n\n cell_metadata slice:")
print(expression_data.cell_metadata)
```

This returns a `CellArrDatasetSlice` object that contains the matrix and metadata `DataFrame`'s along the cell and gene axes.

Users can easily convert these to analysis-ready representations

```{code-cell}
print("as anndata:")
print(expression_data.to_anndata())

print("\n\n as summarizedexperiment:")
print(expression_data.to_summarizedexperiment())
```

---

Check out the [documentation](https://biocpy.github.io/cellarr/api/modules.html) for more details.
