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
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas
import tiledb
from torch import squeeze, Tensor
from pytorch_lightning import LightningDataModule
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .queryutils_tiledb_frame import subset_frame

__author__ = "Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

# Turn off multithreading to allow multiple pytorch dataloader workers
config = tiledb.Config()
config["sm.compute_concurrency_level"] = 1
config["sm.io_concurrency_level"] = 1
config["sm.num_async_threads"] = 1
config["sm.num_reader_threads"] = 1
config["sm.num_tbb_threads"] = 1
config["sm.num_writer_threads"] = 1
config["vfs.num_threads"] = 1


class scDataset(Dataset):
    """A class that extends pytorch :py:class:`~torch.utils.data.Dataset` to enumerate cells and cell labels using
    TileDB."""

    def __init__(
        self,
        data_df: pandas.DataFrame,
        matrix_tdb: tiledb.Array,
        matrix_shape: tuple,
        gene_indices: List[int],
        label_column_name: str,
        study_column_name: str,
        lognorm: bool = True,
        target_sum: float = 1e4,
    ):
        """Initialize a ``scDataset``.

        Args:
            data_df:
                Pandas dataframe of valid cells.

            matrix_tdb:
                TileDB object containing the experimental data, e.g. counts.

            matrix_shape:
                Shape of the counts matrix.

            gene_indices:
                The index of genes to return.

            label_column_name:
                Column name containing cell labels.

            study_column_name:
                Column name containing study information.

            lognorm:
                Whether to return log-normalized expression instead of raw counts.

            target_sum:
                Target sum for log-normalization.
        """

        self.data_df = data_df
        self.matrix_tdb = matrix_tdb
        self.matrix_shape = matrix_shape
        self.gene_indices = gene_indices
        self.lognorm = lognorm
        self.target_sum = target_sum
        self.label_column_name = label_column_name
        self.study_column_name = study_column_name

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        # data, label, study
        cell_idx = self.data_df.index[idx]
        results = self.matrix_tdb.multi_index[cell_idx, :]
        counts = coo_matrix(
            (results["data"], (results["cell_index"], results["gene_index"])),
            shape=self.matrix_shape,
        ).tocsr()
        counts = counts[cell_idx, :]
        counts = counts[:, self.gene_indices]

        X = counts.astype(np.float32)
        if self.lognorm:
            counts_per_cell = counts.sum(axis=1)
            counts_per_cell = np.ravel(counts_per_cell)
            counts_per_cell = counts_per_cell / self.target_sum
            X = X / counts_per_cell[:, None]
            X = csr_matrix(X).log1p()

        X = X.toarray()

        return (
            X,
            self.data_df.loc[cell_idx, self.label_column_name],
            self.data_df.loc[cell_idx, self.study_column_name],
        )

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}("
        output += f"number_of_cells={self.data_df.shape[0]}"
        output += f"number_of_genes={self.matrix_shape[1]}"
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        output += f"number_of_cells: {self.data_df.shape[0]}\n"
        output += f"number_of_genes: {self.matrix_shape[1]}\n"

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
        val_studies: Optional[List[str]] = None,
        gene_order: Optional[List[str]] = None,
        batch_size: int = 1000,
        num_workers: int = 0,
        lognorm: bool = True,
        target_sum: float = 1e4,
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
                Batch size to use.
                Defaults to 1000.

            num_workers:
                The number of worker threads for dataloaders.

            lognorm:
                Whether to return log-normalized expression instead of raw counts.

            target_sum:
                Target sum for log-normalization.

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
        self.gene_order = gene_order
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lognorm = lognorm
        self.target_sum = target_sum

        self.cell_metadata_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.cell_metadata_uri), "r"
        )
        self.gene_annotation_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.gene_annotation_uri), "r"
        )
        self.matrix_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.matrix_uri), "r", config=config
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
            columns=[self.study_column_name, self.label_column_name],
        )

        # limit to labels that exist in at least 2 studies
        study_celltype_counts = (
            self.data_df[[self.study_column_name, self.label_column_name]]
            .drop_duplicates()
            .groupby(self.label_column_name)
            .size()
            .sort_values(ascending=False)
        )
        well_represented_labels = study_celltype_counts[study_celltype_counts > 1].index
        self.data_df = self.data_df[
            self.data_df[self.label_column_name].isin(well_represented_labels)
        ]

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

        self.train_Y = self.data_df[self.label_column_name].values.tolist()
        self.train_study = self.data_df[self.study_column_name].values.tolist()
        self.train_dataset = scDataset(
            data_df=self.data_df,
            matrix_tdb=self.matrix_tdb,
            matrix_shape=self.matrix_shape,
            gene_indices=self.gene_indices,
            label_column_name=self.label_column_name,
            study_column_name=self.study_column_name,
            lognorm=self.lognorm,
            target_sum=self.target_sum,
        )

        self.val_dataset = None
        if self.val_df is not None:
            self.val_Y = self.val_df[self.label_column_name].values.tolist()
            self.val_study = self.val_df[self.study_column_name].values.tolist()
            self.val_dataset = scDataset(
                data_df=self.val_df,
                matrix_tdb=self.matrix_tdb,
                matrix_shape=self.matrix_shape,
                gene_indices=self.gene_indices,
                label_column_name=self.label_column_name,
                study_column_name=self.study_column_name,
                lognorm=self.lognorm,
                target_sum=self.target_sum,
            )

    def __del__(self):
        self.cell_metadata_tdb.close()
        self.gene_annotation_tdb.close()
        self.matrix_tdb.close()

    def get_sampler_weights(
        self, labels: list, studies: Optional[list] = None
    ) -> WeightedRandomSampler:
        """Get weighted random sampler.

        Args:
            dataset:
                Single cell dataset.

        Returns:
            A WeightedRandomSampler object.
        """

        if studies is None:
            class_sample_count = Counter(labels)
            sample_weights = Tensor([1.0 / class_sample_count[t] for t in labels])
        else:
            class_sample_count = Counter(labels)
            study_sample_count = Counter(studies)
            class_sample_count = {
                x: np.log1p(class_sample_count[x] / 1e4) for x in class_sample_count
            }
            study_sample_count = {
                x: np.log1p(study_sample_count[x] / 1e5) for x in study_sample_count
            }
            sample_weights = Tensor(
                [
                    1.0 / class_sample_count[labels[i]] / study_sample_count[studies[i]]
                    for i in range(len(labels))
                ]
            )
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def collate(self, batch):
        """Collate tensors.

        Args:
            batch:
                Batch to collate.

        Returns:
            A Tuple[torch.Tensor, torch.Tensor, list] containing information
            on the collated tensors.
        """

        profiles, labels, studies = tuple(
            map(list, zip(*batch))
        )  # tuple([list(t) for t in zip(*batch)])
        return (
            squeeze(Tensor(np.vstack(profiles))),
            Tensor([self.label2int[label] for label in labels]),  # text to int labels
            studies,
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
            pin_memory=True,
            drop_last=True,
            sampler=self.get_sampler_weights(self.train_Y, self.train_study),
            collate_fn=self.collate,
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
            pin_memory=True,
            drop_last=True,
            sampler=self.get_sampler_weights(self.val_Y, self.val_study),
            collate_fn=self.collate,
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
