from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import cli_method, load_raw_train_eval


def generate_top_correlations(correlation_analysis: CorrelationAnalysis, cutoff=0.7):
    top_correlations_cli = cli_method(
        correlation_analysis.top_correlations, "Calculating feature correlation matrix"
    )
    top_correlations_cli(cutoff=cutoff)


if __name__ == "__main__":
    df = load_raw_train_eval()
    correlation_analysis = CorrelationAnalysis(df)
    generate_top_correlations(correlation_analysis)
