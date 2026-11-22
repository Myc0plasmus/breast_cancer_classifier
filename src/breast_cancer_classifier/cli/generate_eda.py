import pandas as pd
from rich.console import Console

from breast_cancer_classifier.analysis import feature_corelation_matrix
from breast_cancer_classifier.utils import DatasetLoader


def generate_eda() -> None:
    console = Console()

    with console.status("[bold green]Loading datasets..."):
        loader = DatasetLoader(dataset_type="raw")
        train_dataset = loader.load_train()
        eval_dataset = loader.load_eval()

        df = pd.concat([train_dataset, eval_dataset], ignore_index=True)
    console.print("Loading datasets: [green]completed")

    with console.status("[bold green]Calculating feature corelation matix..."):
        feature_corelation_matrix(df, show=True)
    console.print("Calculating feature corelation matrix: [green]completed")


if __name__ == "__main__":
    generate_eda()
