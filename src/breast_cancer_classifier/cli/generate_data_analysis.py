import pandas as pd

from breast_cancer_classifier.analysis import feature_corelation 

from sklearn.datasets import load_breast_cancer

def generate_data_analysis() -> None:
    dataset = load_breast_cancer(as_frame=True)

    df: pd.DataFrame  = dataset["frame"]

    feature_corelation(df)

if __name__ == "__main__":
    generate_data_analysis()


