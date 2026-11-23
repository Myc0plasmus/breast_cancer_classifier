import pandas as pd

from .dataset_loader import DatasetLoader
from .cli_method import cli_method


@cli_method(message="Loading datasets")
def load_raw_train_eval() -> pd.DataFrame:
    loader = DatasetLoader(dataset_type="raw")
    train_dataset = loader.load_train()
    eval_dataset = loader.load_eval()

    df = pd.concat([train_dataset, eval_dataset], ignore_index=True)

    return df
