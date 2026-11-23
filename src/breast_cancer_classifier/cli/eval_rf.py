import pandas as pd
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from breast_cancer_classifier.utils import DatasetLoader, get_tables_path
from breast_cancer_classifier.models import RandomForestWrapped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Random Forest classifier."
    )

    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model to evaluate."
    )

    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Name of the target column in the dataset.",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        default="raw",
        help="Dataset type to load (default: raw).",
    )

    parser.add_argument(
        "--group",
        type=str,
        default="eval",
        choices=["train", "eval", "test"],
        help="Evaluate using given test group (default: eval).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    loader = DatasetLoader(dataset_type=args.dataset_type)

    match args.group:
        case "train":
            dataset = loader.load_train()
            dataset_name = "Train"
        case "eval":
            dataset = loader.load_eval()
            dataset_name = "Eval"
        case "test":
            dataset = loader.load_test()
            dataset_name = "Test"

    if args.target_column not in dataset.columns:
        raise ValueError(
            f"Target column '{args.target_column}' "
            f"not found in {dataset_name.lower()} dataset."
        )

    X = dataset.drop(columns=[args.target_column])
    y_true = dataset[args.target_column]

    if X.empty or len(y_true) == 0:
        raise ValueError(f"{dataset_name} dataset is empty.")

    rf = RandomForestWrapped(model_name=args.model_name)

    if rf.model is None:
        raise ValueError(
            f"Model '{args.model_name}' could not be loaded. Train the model first."
        )

    y_pred = rf.predict(X)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Display results
    print(f"\n{dataset_name} Results")
    print("=" * 35)
    metrics_table = {"Metric": [], "Value": []}

    for metric, value in metrics.items():
        metrics_table["Metric"].append(metric)
        metrics_table["Value"].append(value)
        print(f"{metric:<15}: {value:.4f}")

    print("=" * 35)
    print(f"Samples evaluated: {len(y_true)}")

    df = pd.DataFrame(metrics_table)
    path = get_tables_path(f"{dataset_name}_dataset_evaluation.csv")
    df.to_csv(path)


if __name__ == "__main__":
    main()
