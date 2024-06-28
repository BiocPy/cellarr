# Changelog

## Version 0.2.0

- Thanks to [@tony-kuo](https://github.com/tony-kuo), the package now includes a built-in dataloader for the pytorch-lightning framework,
for single cells expression profiles, training labels, and study labels. The dataloader uniformly samples across training labels and study labels to create a diverse batch of cells.

## Version 0.1.0 - 0.1.3

This is the first release of the package to support both creation and access to large
collection of files based on TileDB.

- Provide a build method to create the tiledb collection from a series of data objects.
- Provides `CellArrDataset` class to query these objects on disk.
- Implements access and coerce methods to interop with other experimental data packages.
