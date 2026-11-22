from pathlib import Path

import pandas as pd

from .get_data_path import get_raw_path


dataset_path_getter_dict = {
    "raw": get_raw_path,
}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested dataset has not been generated yet."""


class DatasetLoader:
    """Utility for loading generated datasets.

    Provides validated access to train, evaluation, and test datasets
    without requiring callers to specify filesystem paths or filenames.

    Args:
        dataset_type (str): Type of dataset to load. Must be one of the
            keys defined in `dataset_path_getter_dict`.

    Raises:
        ValueError: If an unsupported dataset type is provided.
    """

    def __init__(self, dataset_type: str = "raw") -> None:
        if dataset_type not in dataset_path_getter_dict:
            available_types = ", ".join(dataset_path_getter_dict)

            raise ValueError(
                f"Unsupported dataset type '{dataset_type}'. "
                f"Available types: {available_types}."
            )

        self.dataset_type = dataset_type

    def _load_dataset(self, group: str) -> pd.DataFrame:
        """Load a dataset belonging to the specified group.

        Args:
            group (str): Dataset group ('train', 'eval', or 'test').

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            DatasetNotFoundError: If the requested dataset does not exist.
        """
        filename = f"{self.dataset_type}_{group}_dataset.csv"
        path = Path(dataset_path_getter_dict[self.dataset_type](filename))

        if not path.is_file():
            raise DatasetNotFoundError(
                f"Dataset '{filename}' of type '{self.dataset_type}' "
                f"was not found at '{path}'. "
                f"Datasets of type '{self.dataset_type}' "
                "need to be generated first."
            )

        return pd.read_csv(path)

    def load_train(self) -> pd.DataFrame:
        """Load the training dataset.

        Returns:
            pd.DataFrame: Training dataset.
        """
        return self._load_dataset("train")

    def load_eval(self) -> pd.DataFrame:
        """Load the evaluation dataset.

        Returns:
            pd.DataFrame: Evaluation dataset.
        """
        return self._load_dataset("eval")

    def load_test(self) -> pd.DataFrame:
        """Load the test dataset.

        Returns:
            pd.DataFrame: Test dataset.
        """
        return self._load_dataset("test")
