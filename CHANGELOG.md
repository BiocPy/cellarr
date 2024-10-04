# Changelog

## Version 0.2.2

- Provide options to consistently extract cell metadata columns across datasets.
- Update documentation and tests.

## Version 0.2.1

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
