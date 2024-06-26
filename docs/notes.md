---
file_format: mystnb
kernelspec:
  name: python
---

# Usage Notes

While our package provides methods to generate TileDB files, it makes certain assumptions. We will continue to document some of the gotcha's as we run into them. Please review the following for a smooth experience:

#### 1. Experimental Data Consistency

All experimental data objects (either as `AnnData` or H5AD paths) are expected to be fairly consistent:

- **Matrix Location**: If the matrix to use is "counts", all objects must contain this matrix in the `layers` slot, not in `X` or under a different name.
- **Feature IDs/Gene Symbols**: These should be consistent across objects, either as the index or as a column in the `var` dataframe.

#### 2. Cell Metadata

If `cell_metadata` is not provided, the build process scans all files to count the number of cells and creates a simple range index.

#### 3. Sample Information

- Each file is considered a sample, hence a mapping between cells and samples is automatically created.
- The sample information provided must match the number of input files.

#### 4. Handling Metadata Columns with None/NaN Values

For metadata columns containing `None`, `nan`, or `NaN` values:
- It's best to specify `float` as the type of the column
- Even if most values are integers, TileDB may behave unexpectedly with mixed types

For further assistance or clarification, please refer to our documentation or raise an issue on GitHub.
