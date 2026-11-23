from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import load_raw_train_eval
from .correlation import (
    generate_correlation_matrix,
    generate_corr_box_plot,
    generate_target_corr_box_plot,
    generate_top_correlations,
    generate_target_corr_stats,
)


def generate_eda() -> None:
    df = load_raw_train_eval()

    correlation_analysis = CorrelationAnalysis(df)

    generate_correlation_matrix(correlation_analysis)

    generate_corr_box_plot(correlation_analysis)

    generate_target_corr_box_plot(correlation_analysis)

    generate_top_correlations(correlation_analysis)

    generate_target_corr_stats(correlation_analysis)


if __name__ == "__main__":
    generate_eda()
