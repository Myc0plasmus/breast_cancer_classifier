import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from breast_cancer_classifier.utils import get_figure_path


def feature_corelation(df: pd.DataFrame) -> None:
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,           # Display correlation values
        cmap='coolwarm',      # Color palette
        fmt='.2f',            # Two decimal places
        vmin=-1,
        vmax=1,
        center=0,
        square=True
    )

    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()

    path = get_figure_path("feature_correlation.png")
    plt.show() 
