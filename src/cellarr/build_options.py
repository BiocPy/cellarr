import numpy as np

from dataclasses import dataclass

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@dataclass
class CellMetadataOptions:
    """Optional arguments for the ``cell_metadata`` store
    for :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        skip_cell_tiledb:
            Whether to skip generating cell metadata tiledb.
            Defaults to False.

        cell_dim_dtype:
            NumPy dtype for the cell dimension.
            Defaults to np.uint32.

            Note: make sure the number of cells fit
            within the integer limits of unsigned-int32.
    """

    skip_cell_tiledb: bool = False
    cell_dim_dtype: np.dtype = np.uint32


@dataclass
class GeneAnnotationOptions:
    """Optional arguments for the ``gene_annotation`` store
    for :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        var_feature_column:
            Column in ``var`` containing the feature ids (e.g. gene symbols).
            Defaults to the index of the ``var`` slot.

        skip_gene_tiledb:
            Whether to skip generating gene annotation tiledb.
            Defaults to False.

        gene_dim_dtype:
            NumPy dtype for the gene dimension.
            Defaults to np.uint32.

            Note: make sure the number of genes fit
            within the integer limits of unsigned-int32.
    """

    var_feature_column: str = "index"
    skip_gene_tiledb: bool = False
    gene_dim_dtype: np.dtype = np.uint32


@dataclass
class MatrixOptions:
    """Optional arguments for the ``matrix`` store
    for :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        layer_matrix_name:
            Matrix name from ``layers`` slot to add to tiledb.
            Must be consistent across all objects in ``files``.

            Defaults to "counts".

        skip_matrix_tiledb:
            Whether to skip generating matrix tiledb.
            Defaults to False.

        matrix_dim_dtype:
            NumPy dtype for the values in the matrix.
            Defaults to np.uint16.

            Note: make sure the matrix values fit
            within the range limits of unsigned-int16.
    """

    layer_matrix_name: str = "counts"
    skip_matrix_tiledb: bool = False
    matrix_dim_dtype: np.dtype = np.uint16


@dataclass
class SampleMetadataOptions:
    """Optional arguments for the ``sample`` store
    for :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

    Attributes:
        skip_sample_tiledb:
            Whether to skip generating sample tiledb.
            Defaults to False.

        sample_dim_dtype:
            NumPy dtype for the sample dimension.
            Defaults to np.uint32.

            Note: make sure the number of samples fit
            within the integer limits of unsigned-int32.
    """

    skip_sample_tiledb: bool = False
    sample_dim_dtype: np.dtype = np.uint32
