import numpy as np

from dataclasses import dataclass

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@dataclass
class CellMetadataOptions:
    """Optional arguments for the ``cell_metadata`` store for
    :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        skip:
            Whether to skip generating cell metadata tiledb.
            Defaults to False.

        dtype:
            NumPy dtype for the cell dimension.
            Defaults to np.uint32.

            Note: make sure the number of cells fit
            within the integer limits of unsigned-int32.

        tiledb_store_name:
            Name of the tiledb file.
            Defaults to "cell_metadata".
    """

    skip: bool = False
    dtype: np.dtype = np.uint32
    tiledb_store_name: str = "cell_metadata"


@dataclass
class GeneAnnotationOptions:
    """Optional arguments for the ``gene_annotation`` store for
    :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        feature_column:
            Column in ``var`` containing the feature ids (e.g. gene symbols).
            Defaults to the index of the ``var`` slot.

        skip:
            Whether to skip generating gene annotation tiledb.
            Defaults to False.

        dtype:
            NumPy dtype for the gene dimension.
            Defaults to np.uint32.

            Note: make sure the number of genes fit
            within the integer limits of unsigned-int32.

        tiledb_store_name:
            Name of the tiledb file.
            Defaults to "gene_annotation".
    """

    skip: bool = False
    feature_column: str = "index"
    dtype: np.dtype = np.uint32
    tiledb_store_name: str = "gene_annotation"


@dataclass
class MatrixOptions:
    """Optional arguments for the ``matrix`` store for
    :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        matrix_name:
            Matrix name from ``layers`` slot to add to tiledb.
            Must be consistent across all objects in ``files``.

            Defaults to "counts".

        consolidate_duplicate_gene_func:
            Function to consolidate when the AnnData object contains
            multiple rows with the same feature id or gene symbol.

            Defaults to :py:func:`sum`.

        skip:
            Whether to skip generating matrix tiledb.
            Defaults to False.

        dtype:
            NumPy dtype for the values in the matrix.
            Defaults to np.uint16.

            Note: make sure the matrix values fit
            within the range limits of unsigned-int16.

        tiledb_store_name:
            Name of the tiledb file.
            Defaults to `matrix`.
    """

    skip: bool = False
    consolidate_duplicate_gene_func: callable = sum
    matrix_name: str = "counts"
    dtype: np.dtype = np.uint16
    tiledb_store_name: str = "counts"


@dataclass
class SampleMetadataOptions:
    """Optional arguments for the ``sample`` store for :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        skip:
            Whether to skip generating sample tiledb.
            Defaults to False.

        dtype:
            NumPy dtype for the sample dimension.
            Defaults to np.uint32.

            Note: make sure the number of samples fit
            within the integer limits of unsigned-int32.

        tiledb_store_name:
            Name of the tiledb file.
            Defaults to "sample_metadata".
    """

    skip: bool = False
    dtype: np.dtype = np.uint32
    tiledb_store_name: str = "sample_metadata"
