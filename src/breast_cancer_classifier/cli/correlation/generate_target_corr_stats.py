from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import cli_method, load_raw_train_eval


def generate_target_corr_stats(correlation_analysis: CorrelationAnalysis):
    target_corr_stats_cli = cli_method(
        correlation_analysis.target_correlation_statistics,
        "Calculating feature correlation matrix",
    )
    target_corr_stats_cli()


if __name__ == "__main__":
    df = load_raw_train_eval()
    correlation_analysis = CorrelationAnalysis(df)
    generate_target_corr_stats(correlation_analysis)
