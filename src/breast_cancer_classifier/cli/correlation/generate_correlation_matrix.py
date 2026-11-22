from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import cli_method, load_raw_train_eval


def generate_correlation_matrix(
    correlation_analysis: CorrelationAnalysis, show: bool = True
):
    correlation_matrix_cli = cli_method(
        correlation_analysis.matrix, "Calculating feature correlation matrix"
    )
    correlation_matrix_cli(show=show)


if __name__ == "__main__":
    df = load_raw_train_eval()
    correlation_analysis = CorrelationAnalysis(df)
    generate_correlation_matrix(correlation_analysis)
