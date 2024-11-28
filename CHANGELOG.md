# Changelog

## Version 0.3.0

This version introduces major improvements to matrix handling, storage, and performance, including support for multiple matrices in H5AD/AnnData workflows and optimizations for ingestion and querying.

**Support for multiple matrices**:
  - Both `build_cellarrdataset` and `CellArrDataset` now support multiple matrices. During ingestion, a TileDB group called `"assays"` is created to store all matrices, along with group-level metadata.

This may introduce breaking changes with the default parameters based on how these classes are used. Previously to build the TileDB files:

```python
dataset = build_cellarrdataset(
    output_path=tempdir,
    files=[adata1, adata2],
    matrix_options=MatrixOptions(matrix_name="counts", dtype=np.int16),
    num_threads=2,
)
```

Now you may provide a list of matrix options for each layers in the files.

```python
dataset = build_cellarrdataset(
    output_path=tempdir,
    files=[adata1, adata2],
    matrix_options=[
        MatrixOptions(matrix_name="counts", dtype=np.int16),
        MatrixOptions(matrix_name="log-norm", dtype=np.float32),
    ],
    num_threads=2,
)
```

Querying follows a similar structure:
```python
cd = CellArrDataset(
    dataset_path=tempdir,
    assay_tiledb_group="assays",
    assay_uri=["counts", "log-norm"]
)
```
`assay_uri` is relative to `assay_tiledb_group`. For backwards compatibility, `assay_tiledb_group` can be an empty string.

**Parallelized ingestion**:
The build process now uses `num_threads` to ingest matrices concurrently. Two new columns in the sample metadata, `cellarr_sample_start_index` and `cellarr_sample_end_index`, track sample offsets, improving matrix processing.
  - Note: The process pool uses the `spawn` method on UNIX systems, which may affect usage on windows machines.

**TileDB query condition fixes**:
Fixed a few issues with fill values represented as bytes (seems to be common when ascii is used as the column type) and in general filtering operations on TileDB Dataframes.

**Index remapping**:
Improved remapping of indices from sliced TileDB arrays for both dense and sparse matrices. This is not a user facing function but an internal slicing operation.

**Get a sample**:
Added a method to access all cells for a particular sample. you can either provide an index or a sample id.

```python
sample_1_slice = cd.get_cells_for_sample(0)
```

Other updates to documentation, tutorials, the README, and additional tests.

## Version 0.2.4 - 0.2.5

- Provide options to extract an expected set of cell metadata columns across datasets.
- Update documentation and tests.

## Version 0.2.1 - 0.2.3

* Implement dunder methods `__len__`,  `__repr__` and `__str__` for the `CellArrDatasetSlice` class
* Add property `shape` to the same class
* Improve package load time


## Version 0.2.0

- Thanks to [@tony-kuo](https://github.com/tony-kuo), the package now includes a built-in dataloader for the pytorch-lightning framework,
for single cells expression profiles, training labels, and study labels. The dataloader uniformly samples across training labels and study labels to create a diverse batch of cells.

- Minor fixes for CSV to TileDB conversion for the `cell_metadata` object.

## Version 0.1.0 - 0.1.3

This is the first release of the package to support both creation and access to large
collection of files based on TileDB.

- Provide a build method to create the TileDB collection from a series of data objects.
- Provides `CellArrDataset` class to query these objects on disk.
- Implements access and coerce methods to interop with other experimental data packages.
