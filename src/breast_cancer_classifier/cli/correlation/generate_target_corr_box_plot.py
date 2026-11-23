from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import cli_method, load_raw_train_eval


def generate_target_corr_box_plot(
    correlation_analysis: CorrelationAnalysis, show: bool = True
):
    target_correlation_box_plot_cli = cli_method(
        correlation_analysis.target_box_plot, "Calculating feature correlation matrix"
    )
    target_correlation_box_plot_cli(show=show)


if __name__ == "__main__":
    df = load_raw_train_eval()
    correlation_analysis = CorrelationAnalysis(df)
    generate_target_corr_box_plot(correlation_analysis)
