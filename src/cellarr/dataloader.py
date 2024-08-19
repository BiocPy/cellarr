"""A dataloader using TileDB files in the pytorch-lightning framework.

This class provides a dataloader using the generated TileDB files built using the
:py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

Example:

    .. code-block:: python

        from cellarr.dataloader import DataModule

        datamodule = DataModule(
            dataset_path="/path/to/cellar/dir",
            cell_metadata_uri="cell_metadata",
            gene_annotation_uri="gene_annotation",
            matrix_uri="counts",
            val_studies=["test3"],
            label_column_name="label",
            study_column_name="study",
            batch_size=100,
            lognorm=True,
            target_sum=1e4,
        )

        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))
        data, labels, studies = batch
        print(data, labels, studies)
"""

import os
from typing import List, Optional

import numpy as np
import pandas
import tiledb
from torch import Tensor
from pytorch_lightning import LightningDataModule
import random
from scipy.sparse import coo_matrix, diags
from torch.utils.data import DataLoader, Dataset

from .queryutils_tiledb_frame import subset_frame

__author__ = "Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class scDataset(Dataset):
    """A class that extends pytorch :py:class:`~torch.utils.data.Dataset` to enumerate cells and cell metadata using
    TileDB."""

    def __init__(
        self,
        data_df: pandas.DataFrame,
        int2sample: dict,
        sample2cells: dict,
        sample_size: int,
        sampling_by_class: bool = False,
    ):
        """Initialize a ``scDataset``.

        Args:
            data_df:
                Pandas dataframe of valid cells.

            int2sample:
                A mapping of sample index to sample id.

            sample2cells:
                A mapping of sample id to cell indices.

            sample_size:
                Number of cells one sample.

            sampling_by_class:
                Sample based on class counts, where sampling weight is inversely proportional to count.
                Defaults to False.
        """

        self.data_df = data_df
        self.int2sample = int2sample
        self.sample2cells = sample2cells
        self.sample_size = sample_size
        self.sampling_by_class = sampling_by_class

    def __len__(self):
        return len(self.int2sample)

    def __getitem__(self, idx):
        sample = self.int2sample[idx]
        cell_idx = self.sample2cells[sample]
        if len(cell_idx) < self.sample_size:
            cell_idx = random.sample(self.sample2cells[sample], len(cell_idx))
        else:
            if self.sampling_by_class:
                sample_df = self.data_df.loc[self.sample2cells[sample], :].copy()
                sample_df = sample_df.sample(
                    n=self.sample_size, weights="sample_weight"
                )
                cell_idx = sample_df.index.tolist()
            else:
                cell_idx = random.sample(self.sample2cells[sample], self.sample_size)

        return self.data_df.loc[cell_idx].copy()

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}("
        output += f"number_of_cells={self.data_df.shape[0]}"
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        output += f"number_of_cells: {self.data_df.shape[0]}\n"

        return output


class DataModule(LightningDataModule):
    """A class that extends a pytorch-lightning :py:class:`~pytorch_lightning.LightningDataModule` to create pytorch
    dataloaders using TileDB.

    The dataloader uniformly samples across training labels and study labels to create a diverse batch of cells.
    """

    def __init__(
        self,
        dataset_path: str,
        cell_metadata_uri: str = "cell_metadata",
        gene_annotation_uri: str = "gene_annotation",
        matrix_uri: str = "counts",
        label_column_name: str = "celltype_id",
        study_column_name: str = "study",
        sample_column_name: str = "cellarr_sample",
        val_studies: Optional[List[str]] = None,
        gene_order: Optional[List[str]] = None,
        batch_size: int = 100,
        sample_size: int = 100,
        num_workers: int = 1,
        lognorm: bool = True,
        target_sum: float = 1e4,
        sparse: bool = False,
        sampling_by_class: bool = False,
        remove_singleton_classes: bool = False,
        nan_string: str = "nan",
    ):
        """Initialize a ``DataModule``.

        Args:
            dataset_path:
                Path to the directory containing the TileDB stores.
                Usually the ``output_path`` from the
                :py:func:`~cellarr.build_cellarrdataset.build_cellarrdataset`.

            cell_metadata_uri:
                Relative path to cell metadata store.

            gene_annotation_uri:
                Relative path to gene annotation store.

            matrix_uri:
                Relative path to matrix store.

            label_column_name:
                Column name in `cell_metadata_uri` containing cell labels.

            study_column_name:
                Column name in `cell_metadata_uri` containing study information.

            val_studies:
                List of studies to use for validation and test.
                If None, all studies are used for training.

            gene_order:
                List of genes to subset to from the gene space.
                If None, all genes from the `gene_annotation` are used for training.

            batch_size:
                Batch size to use, corresponding to the number of samples in a mini-batch.
                Defaults to 100.

            sample_size:
                Size of each sample use in a mini-batch, corresponding to the number of cells in a sample.
                Defaults to 100.

            num_workers:
                The number of worker threads for dataloaders.
                Defaults to 1.

            lognorm:
                Whether to return log-normalized expression instead of raw counts.

            target_sum:
                Target sum for log-normalization.

            sparse:
                Whether to return a sparse tensor.
                Defaults to False.

            sampling_by_class:
                Sample based on class counts, where sampling weight is inversely proportional to count.
                If False, use random sampling. Defaults to False.

            remove_singleton_classes:
                Exclude cells with classes that exist in only one sample.
                Defaults to False.

            nan_string:
                A string representing NaN.
                Defaults to "nan".
        """

        super().__init__()
        self.dataset_path = dataset_path
        self.cell_metadata_uri = cell_metadata_uri
        self.gene_annotation_uri = gene_annotation_uri
        self.matrix_uri = matrix_uri
        self.val_studies = val_studies
        self.label_column_name = label_column_name
        self.study_column_name = study_column_name
        self.sample_column_name = sample_column_name
        self.gene_order = gene_order
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_workers = num_workers
        self.lognorm = lognorm
        self.target_sum = target_sum
        self.sparse = sparse
        self.sampling_by_class = sampling_by_class
        self.remove_singleton_classes = remove_singleton_classes

        self.cell_metadata_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.cell_metadata_uri), "r"
        )
        self.gene_annotation_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.gene_annotation_uri), "r"
        )
        self.matrix_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.matrix_uri), "r"
        )

        self.matrix_shape = (
            self.cell_metadata_tdb.nonempty_domain()[0][1] + 1,
            self.gene_annotation_tdb.nonempty_domain()[0][1] + 1,
        )

        # limit to cells with labels
        query_condition = f"{self.label_column_name} != '{nan_string}'"
        self.data_df = subset_frame(
            self.cell_metadata_tdb,
            query_condition,
            columns=[
                self.study_column_name,
                self.sample_column_name,
                self.label_column_name,
            ],
        )

        # concat study and sample in the case there are duplicate sample names
        self.sampleID = "study::::sample"
        self.data_df[self.sampleID] = (
            self.data_df[self.study_column_name]
            + "::::"
            + self.data_df[self.sample_column_name]
        )

        if self.remove_singleton_classes:
            # limit to celltypes that exist in at least 2 samples
            celltype_counts = (
                self.data_df[[self.sampleID, self.label_column_name]]
                .drop_duplicates()
                .groupby(self.label_column_name)
                .size()
                .sort_values(ascending=False)
            )
            well_represented_labels = celltype_counts[celltype_counts > 1].index
            self.data_df = self.data_df[
                self.data_df[self.label_column_name].isin(well_represented_labels)
            ]

        if self.sampling_by_class:
            # get sampling weights based on class
            class_sample_count = self.data_df[self.label_column_name].value_counts()
            class_sample_count = {
                x: np.log1p(class_sample_count[x] / 1e4)
                for x in class_sample_count.index
            }
            self.data_df["sample_weight"] = self.data_df[self.label_column_name].apply(
                lambda x: 1.0 / class_sample_count[x]
            )

        self.val_df = None
        if self.val_studies is not None:
            # split out validation studies
            self.val_df = self.data_df[
                self.data_df[self.study_column_name].isin(self.val_studies)
            ]
            self.data_df = self.data_df[
                ~self.data_df[self.study_column_name].isin(self.val_studies)
            ]
            # limit validation celltypes to those in the training data
            self.val_df = self.val_df[
                self.val_df[self.label_column_name].isin(
                    self.data_df[self.label_column_name].unique()
                )
            ]

        print(f"Training data size: {self.data_df.shape}")
        if self.val_df is not None:
            print(f"Validation data size: {self.val_df.shape}")

        self.class_names = set(self.data_df[self.label_column_name].values)
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {value: key for key, value in self.label2int.items()}

        genes = (
            self.gene_annotation_tdb.query(attrs=["cellarr_gene_index"])
            .df[:]["cellarr_gene_index"]
            .tolist()
        )
        if self.gene_order is not None:
            self.gene_indices = []
            for x in self.gene_order:
                try:
                    self.gene_indices.append(genes.index(x))
                except NameError:
                    print(f"Gene not found: {x}")
                    pass
        else:
            self.gene_indices = [i for i in range(len(genes))]

        gp = self.data_df.groupby(self.sampleID)
        self.train_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
        self.train_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
        self.data_df["label_int"] = self.data_df[self.label_column_name].map(
            self.label2int
        )
        self.train_dataset = scDataset(
            data_df=self.data_df,
            int2sample=self.train_int2sample,
            sample2cells=self.train_sample2cells,
            sample_size=self.sample_size,
            sampling_by_class=self.sampling_by_class,
        )

        self.val_dataset = None
        if self.val_df is not None:
            gp = self.val_df.groupby(self.sampleID)
            self.val_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
            self.val_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
            self.val_df["label_int"] = self.val_df[self.label_column_name].map(
                self.label2int
            )
            self.val_dataset = scDataset(
                data_df=self.val_df,
                int2sample=self.val_int2sample,
                sample2cells=self.val_sample2cells,
                sample_size=self.sample_size,
                sampling_by_class=self.sampling_by_class,
            )

    def __del__(self):
        self.cell_metadata_tdb.close()
        self.gene_annotation_tdb.close()
        self.matrix_tdb.close()

    def collate(self, batch):
        """Collate tensors.

        Args:
            batch:
                Batch to collate.

        Returns:
            tuple
                A Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray] containing information
                corresponding to [input, label, study, sample]
        """

        df = pandas.concat(batch)
        cell_idx = df.index.tolist()

        results = self.matrix_tdb.multi_index[cell_idx, :]
        counts = coo_matrix(
            (results["data"], (results["cell_index"], results["gene_index"])),
            shape=self.matrix_shape,
        ).tocsr()
        counts = counts[cell_idx, :]
        counts = counts[:, self.gene_indices]

        X = counts.astype(np.float32)
        if self.lognorm:
            # normalize to target sum
            row_sums = np.ravel(X.sum(axis=1))  # row sums as a 1D array
            # avoid division by zero by setting zero sums to one (they will remain zero after normalization)
            row_sums[row_sums == 0] = 1
            # create a sparse diagonal matrix with the inverse of the row sums
            inv_row_sums = diags(1 / row_sums).tocsr()
            # normalize the rows to sum to 1
            normalized_matrix = inv_row_sums.dot(X)
            # scale the rows sum to target_sum
            X = normalized_matrix.multiply(datamodule.target_sum)
            X = X.log1p()

        X = Tensor(X.toarray())
        if self.sparse:
            X = X.to_sparse_csr()

        return (
            X,
            Tensor(df["label_int"].values),
            df[self.study_column_name].values,
            df[self.sample_column_name].values,
        )

    def train_dataloader(self) -> DataLoader:
        """Load the training dataset.

        Returns:
            A DataLoader object containing the training dataset.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def val_dataloader(self) -> DataLoader:
        """Load the validation dataset.

        Returns:
            A DataLoader object containing the validation dataset.
        """

        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}("
        output += f"number_of_training_cells={self.data_df.shape[0]}"
        if self.val_df is not None:
            output += f", number_of_validation_cells={self.val_df.shape[0]}"
        else:
            output += ", number_of_validation_cells=0"
        output += f", at path={self.dataset_path}"
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        output += f"number_of_training_cells: {self.data_df.shape[0]}\n"
        if self.val_df is not None:
            output += f"number_of_validation_cells: {self.val_df.shape[0]}\n"
        else:
            output += "number_of_validation_cells: 0\n"
        output += f"number_of_genes: {len(self.gene_indices)}\n"
        output += f"path: '{self.dataset_path}'\n"

        return output
