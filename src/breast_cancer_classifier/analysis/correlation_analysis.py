import matplotlib

matplotlib.use("QtAgg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from breast_cancer_classifier.utils import (
    get_figure_path,
    get_tables_path,
)


class CorrelationAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.corr_matrix = df.corr()
        self.df = df

    def matrix(self, show=False) -> None:
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            self.corr_matrix,
            annot=False,
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

        path = get_figure_path("feature_correlation_matrix.png")
        plt.savefig(path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def box_plot(self, show=False) -> None:
        # Extract unique pairs, excluding self-correlations
        values = self.corr_matrix.where(
            pd.DataFrame(
                True, index=self.corr_matrix.index, columns=self.corr_matrix.columns
            ).where(
                lambda mask: mask.index.to_series().values[:, None]
                != mask.columns.to_series().values[None, :],
                False,
            )
        )

        values = self.corr_matrix.values[
            __import__("numpy").triu_indices(len(self.corr_matrix), k=1)
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(y=values, ax=ax)

        ax.set_title("Distribution of Correlation Values")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1)

        plt.tight_layout()

        path = get_figure_path("correlation_box_plot.png")
        plt.savefig(path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def top_correlations(self, cutoff=0.7) -> pd.DataFrame:
        # Extract unique feature pairs, excluding diagonal
        upper_triangle = self.corr_matrix.where(
            __import__("numpy")
            .triu(__import__("numpy").ones(self.corr_matrix.shape), k=1)
            .astype(bool)
        )

        correlations = upper_triangle.stack().reset_index()

        correlations.columns = ["param1", "param2", "correlation value"]

        # Keep correlations whose absolute value meets the cutoff
        correlations = correlations[correlations["correlation value"].abs() >= cutoff]

        correlations["abs_correlation"] = correlations["correlation value"].abs()

        correlations = correlations.sort_values(
            by="abs_correlation", ascending=False
        ).drop(columns="abs_correlation")

        path = get_tables_path("top_correlation_values.csv")
        correlations.to_csv(path, index=False)

        return correlations

    def top_target_corr(self, cutoff=0.7) -> pd.DataFrame:
        correlations = self.corr_matrix["target"].drop(index="target")

        correlations = correlations[correlations.abs() >= cutoff]

        correlations = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )

        result = correlations.rename("correlation value").reset_index()
        result.columns = ["feature", "correlation value"]

        path = get_tables_path("top_target_corr_values.csv")
        result.to_csv(path, index=False)

        return result

    def target_box_plot(self, show=False) -> None:
        target_correlations = self.corr_matrix["target"].drop("target")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(y=target_correlations.values, ax=ax)

        ax.set_title("Distribution of Feature Correlations with Target")
        ax.set_ylabel("Correlation with Target")
        ax.set_ylim(-1, 1)

        plt.tight_layout()

        path = get_figure_path("target_correlation_box_plot.png")
        plt.savefig(path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def target_correlation_statistics(self) -> pd.DataFrame:
        target_correlations = self.corr_matrix["target"].drop("target")

        statistics = pd.DataFrame(
            {
                "statistic": ["mean", "standard deviation", "min", "max"],
                "value": [
                    target_correlations.mean(),
                    target_correlations.std(),
                    target_correlations.min(),
                    target_correlations.max(),
                ],
            }
        )

        path = get_tables_path("target_correlation_statistics.csv")
        statistics.to_csv(path, index=False)

        return statistics
