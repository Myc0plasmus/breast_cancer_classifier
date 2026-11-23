from rich.console import Console

from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import load_raw_train_eval
from .correlation import (
    generate_correlation_matrix,
    generate_corr_box_plot,
    generate_target_corr_box_plot,
)


def generate_eda() -> None:
    console = Console()
    df = load_raw_train_eval()

    correlation_analysis = CorrelationAnalysis(df)

    generate_correlation_matrix(correlation_analysis)

    generate_corr_box_plot(correlation_analysis)

    generate_target_corr_box_plot(correlation_analysis)

    with console.status("[bold green]Calculating parameters with top correlations..."):
        correlation_analysis.top_correlations()
    console.print("Calculating parameters with top correlations: [green]completed")


if __name__ == "__main__":
    generate_eda()
