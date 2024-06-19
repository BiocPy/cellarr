[![PyPI-Server](https://img.shields.io/pypi/v/cellarr.svg)](https://pypi.org/project/cellarr/)
![Unit tests](https://github.com/BiocPy/cellarr/actions/workflows/pypi-test.yml/badge.svg)

# Cell Arrays

Cell Arrays is a Python package that provides a TileDB-backed store for large collections of genomic experimental data, such as millions of cells across multiple single-cell experiment objects.

The `CellArrDataset` is designed to store single-cell RNA-seq
datasets but can be generalized to store any 2-dimensional experimental data.

## Install

To get started, install the package from [PyPI](https://pypi.org/project/cellarr/)

```bash
pip install cellarr
```

## Usage

### Build a `CellArrDataset`

Building a `CellArrDataset` generates three TileDB files in the specified output directory:

- `gene_metadata`: Contains feature annotations.
- `cell_metadata`: Contains cell or sample metadata.
- `matrix`: A TileDB-backed sparse array containing expression vectors.

The TileDB matrix file is stored in a cell X gene orientation. This orientation
is chosen because the fastest-changing dimension as new files are added to the
collection is usually the cells rather than genes.

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
     files=[adata1, adata2],
     matrix_dim_dtype=np.float32
)
```

The build process usually involves 3 steps:

1. **Scan the Collection**: Scan the entire collection of files to create
a unique set of gene symbols. Store this gene set as the
`gene_metadata` TileDB file.
2. **Store Cell Metadata**: Store cell metadata as the
`cell_metadata` TileDB file.
3. **Remap and Orient Data**: For each dataset in the collection,
remap and orient the gene dimension using the gene set from Step 1.
This step ensures consistency in gene measurement and order, even if
some genes are unmeasured or ordered differently in the original experiments.

***Note: The objects to build the `CellArrDataset` are expected to be fairly consistent, especially along the feature dimension.
if these are `AnnData` or `H5AD`objects, all objects must contain an index (in the `var` slot) specifying the gene symbols.***

Check out the [documentation](https://biocpy.github.io/cellarr/tutorial.html) for more details.

----

#### TODO: This following section does not work yet

Users have the option to reuse the `dataset` object retuned when building the dataset or by creating a `CellArrDataset` object by initializing it to the path where the files were created.

```python
# Create a CellArrDataset object from the existing dataset
dataset = CellArrDataset(dataset_path=tempdir)

# Query data from the dataset
expression_data = dataset[10, ["gene1", "gene10", "gene500"]]
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see <https://pyscaffold.org/>.
