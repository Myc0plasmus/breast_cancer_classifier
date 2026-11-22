from rich.console import Console

from breast_cancer_classifier.data_processing import split_raw_dataset
from breast_cancer_classifier.utils import generate_data, get_raw_path


def generate_split_datasets() -> None:
    console = Console()

    dataset_names = [
        "train",
        "eval",
        "test",
    ]

    generate_data()

    for name, dataset in zip(dataset_names, split_raw_dataset()):
        filename = f"{name}_dataset.csv"
        path = get_raw_path(filename)
        with console.status(f"[bold green]Saving {name} dataset to {path} ..."):
            dataset.to_csv(path)
        console.print(f"Saving {name} dataset to {path}: [green]completed")


if __name__ == "__main__":
    generate_split_datasets()
