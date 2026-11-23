from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import cli_method, load_raw_train_eval


def generate_top_target_corr(correlation_analysis: CorrelationAnalysis, cutoff=0.7):
    top_target_corr_cli = cli_method(
        correlation_analysis.top_target_corr, "Calculating top correlation values"
    )
    return top_target_corr_cli(cutoff=cutoff)


if __name__ == "__main__":
    df = load_raw_train_eval()
    correlation_analysis = CorrelationAnalysis(df)
    top_target_corr = generate_top_target_corr(correlation_analysis)
    print(top_target_corr)
