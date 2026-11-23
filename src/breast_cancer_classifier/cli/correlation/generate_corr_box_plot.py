from breast_cancer_classifier.analysis import CorrelationAnalysis
from breast_cancer_classifier.utils import cli_method, load_raw_train_eval


def generate_corr_box_plot(
    correlation_analysis: CorrelationAnalysis, show: bool = True
):
    correlation_box_plot_cli = cli_method(
        correlation_analysis.box_plot, "Calculating feature correlation box-plot"
    )
    correlation_box_plot_cli(show=show)


if __name__ == "__main__":
    df = load_raw_train_eval()
    correlation_analysis = CorrelationAnalysis(df)
    generate_corr_box_plot(correlation_analysis)
