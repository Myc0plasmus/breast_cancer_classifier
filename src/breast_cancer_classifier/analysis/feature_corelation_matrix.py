import matplotlib

matplotlib.use("QtAgg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from breast_cancer_classifier.utils import get_figure_path


def feature_corelation_matrix(df: pd.DataFrame, show=False) -> None:
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot_kws={"size": 8, "ha": "center", "va": "center"},
        linewidths=1,
        linecolor="white",
    )

    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()

    path = get_figure_path("feature_correlation.png")
    plt.savefig(path)

    if show:
        plt.show()
