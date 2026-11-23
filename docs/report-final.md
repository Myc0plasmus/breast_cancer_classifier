
Imię|Nazwisko|nr. indeksu
-|-|-
Natan|Jabłoński|155621

# Analiza eksploracyjna

## Analiza korelacji
Poniższa sekcja przedstawia analizę korelacji na rekordach z surowych danych ze zbioru treningowego i walidacyjnego.

<!-- ### Kod źródłowy -->
<!-- Poniższa analiza została wykonana za pomocą poniższego kodu źródłowego -->
<!-- ```python -->
<!-- import matplotlib

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
 -->
<!-- ``` -->

### Macierz korelacji

![](artifacts/figures/feature_correlation_matrix.png)

### Wykres pudełkowy wartości korelacji


![](artifacts/figures/correlation_box_plot.png)

Korelacja cech do targetu jest głównie ujemna, aczkolwiek dla paru cech jest wysoka. Mimo to przez fakt, że target jest binarny to lepiej uważać z poleganiem na korelacji do targetu pod kątem dobierania cech.

### Wykres pudełkowy wartości korelacji do targetu

![](artifacts/figures/target_correlation_box_plot.png)

Brak dużej ilość wartości odstających

### Największe wartości korelacji do targetu

| feature | correlation value |
| --- | --- |
| worst concave points | -0.7991281963528148 |
| worst perimeter | -0.7872884119600874 |
| mean concave points | -0.782891170520241 |
| worst radius | -0.7801540578479679 |
| worst area | -0.7439604444160878 |
| mean perimeter | -0.7438100246908919 |
| mean radius | -0.7305304502630511 |
| mean area | -0.7141351046841047 |

### Statystyki wartości korelacji do targetu

| statistic | value |
| --- | --- |
| mean | -0.4437862915802145 |
| standard deviation | 0.29860218709333536 |
| min | -0.7991281963528148 |
| max | 0.2843530087799683 |

Te wartości pokazują że można w miarę zaufać temu, że wysoka korelacja do niektórych cech.

# Wyniki modelu

## Random Forest

### Ewaluacja datasetu treningowego
|  | Metric | Value |
| --- | --- | --- |
| 0 | Accuracy | 1.0 |
| 1 | Precision | 1.0 |
| 2 | Recall | 1.0 |
| 3 | F1 Score | 1.0 |

### Ewaluacja datasetu ewalucyjnego
|  | Metric | Value |
| --- | --- | --- |
| 0 | Accuracy | 0.9824561403508771 |
| 1 | Precision | 0.9829302987197724 |
| 2 | Recall | 0.9824561403508771 |
| 3 | F1 Score | 0.9823623542652152 |

### Ewaluacja datasetu testowego
|  | Metric | Value |
| --- | --- | --- |
| 0 | Accuracy | 0.9298245614035088 |
| 1 | Precision | 0.9298245614035088 |
| 2 | Recall | 0.9298245614035088 |
| 3 | F1 Score | 0.9298245614035088 |

### Wyniki ewaluacji-krzyżowej
Do ewaluacji wykorzystanie stratified K-fold z k=20:

|  | Metric | Mean | STD |
| --- | --- | --- | --- |
| 0 | Accuracy | 0.9629446640316205 | 0.042978360902614796 |
| 1 | Precision | 0.9655401995347648 | 0.040548783743800845 |
| 2 | Recall | 0.9629446640316205 | 0.042978360902614796 |
| 3 | F1 | 0.9630330397459869 | 0.04263517076389311 |

### Krzywa uczenia

![](artifacts/figures/my_random_forest_learning_curve.png)




<!-- ## UMAP -->
<!-- ![[../artifacts/figures/cor_inter_params.png]] -->








