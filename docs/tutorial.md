---
file_format: mystnb
kernelspec:
  name: python
---

# Cell Arrays

Cell Arrays is a Python package that provides a TileDB-backed store for large collections of genomic experimental data, such as millions of cells across multiple single-cell experiment objects.

## Usage

### Create the `CellArrDataset`

Creating a CellArrDataset generates three TileDB files in the specified output directory:

- `gene_metadata`: Contains feature annotations.
- `cell_metadata`: Contains cell or sample metadata.
- `matrix`: A TileDB-backed sparse array containing expression vectors.

***Note: Currently only supports either paths to H5AD or `AnnData` objects***

To build a `CellArrDataset` from a collection of `H5AD` or `AnnData` objects:

```python
import anndata
import numpy as np
import tempfile
from cellarr import build_cellarrdataset, CellArrDataset

# Create a temporary directory
tempdir = tempfile.mkdtemp()

# Read AnnData objects
adata1 = anndata.read_h5ad("path/to/object1.h5ad")
# or just provide the path
adata2 = "path/to/object2.h5ad"

# Build CellArrDataset
dataset = build_cellarrdataset(
     output_path=tempdir,
     h5ad_or_adata=[adata1, adata2],
     matrix_dim_dtype=np.float32
)
```
----

#### TODO: This following section does not work yet.

Users have the option to reuse the `dataset` object retuned when building the dataset or by creating a `CellArrDataset` object by initializng it to the path where the files were created.

```python
# Create a CellArrDataset object from the existing dataset
dataset = CellArrDataset(dataset_path=tempdir)

# Query data from the dataset
expression_data = dataset[10, ["gene1", "gene10", "gene500"]]
```
