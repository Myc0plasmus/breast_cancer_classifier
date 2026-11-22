import pandas as pd

from typing import cast
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split


def split_raw_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = cast(Bunch, load_breast_cancer(as_frame=True))
    df = cast(pd.DataFrame, dataset.frame)

    train_dataset, temp_dataset = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["target"],
        ),
    )

    eval_dataset, test_dataset = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            temp_dataset,
            test_size=0.5,
            random_state=42,
            stratify=temp_dataset["target"],
        ),
    )

    return train_dataset, eval_dataset, test_dataset
