import pandas as pd
from rich.console import Console

from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import DatasetLoader


def generate_eda() -> None:
    console = Console()

    with console.status("[bold green]Loading datasets..."):
        loader = DatasetLoader(dataset_type="raw")
        train_dataset = loader.load_train()
        eval_dataset = loader.load_eval()

        df = pd.concat([train_dataset, eval_dataset], ignore_index=True)
    console.print("Loading datasets: [green]completed")

    correlation_analysis = CorrelationAnalysis(df)

    with console.status("[bold green]Calculating feature correlation matrix..."):
        correlation_analysis.matrix(show=True)
    console.print("Calculating feature correlation matrix: [green]completed")

    with console.status("[bold green]Calculating feature correlation boxplot..."):
        correlation_analysis.box_plot(show=True)
    console.print("Calculating feature correlation boxplot: [green]completed")

    with console.status("[bold green]Calculating target correlation boxplot..."):
        correlation_analysis.target_box_plot(show=True)
    console.print("Calculating feature correlation target: [green]completed")

    with console.status("[bold green]Calculating parameters with top correlations..."):
        correlation_analysis.top_correlations()
    console.print("Calculating parameters with top correlations: [green]completed")


if __name__ == "__main__":
    generate_eda()
