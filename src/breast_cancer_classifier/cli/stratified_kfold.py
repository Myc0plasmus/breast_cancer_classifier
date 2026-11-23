import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from breast_cancer_classifier.utils import DatasetLoader, get_tables_path
from breast_cancer_classifier.models import RandomForestWrapped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stratified K-fold cross-validation."
    )

    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the Random Forest model."
    )

    parser.add_argument(
        "--target-column", type=str, required=True, help="Name of the target column."
    )

    parser.add_argument(
        "--dataset-type", type=str, default="raw", help="Dataset type (default: raw)."
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest.",
    )

    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility."
    )

    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main():
    args = parse_args()

    # Load training dataset
    loader = DatasetLoader(dataset_type=args.dataset_type)
    dataset = loader.load_train()

    if args.target_column not in dataset.columns:
        raise ValueError(f"Target column '{args.target_column}' not found.")

    X = dataset.drop(columns=[args.target_column])
    y = dataset[args.target_column]

    if X.empty or len(y) == 0:
        raise ValueError("Training dataset is empty.")

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    class_counts = y.value_counts()

    if class_counts.min() < args.n_splits:
        raise ValueError(
            f"Each class must have at least {args.n_splits} samples. "
            f"Smallest class has {class_counts.min()} samples."
        )

    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.random_state
    )

    rf = RandomForestWrapped(model_name=args.model_name)

    results = []

    print(f"\nStratified {args.n_splits}-Fold Cross-Validation")
    print("=" * 65)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        rf.train(X_train, y_train, n_estimators=args.n_estimators, overwrite_model=True)

        y_pred = rf.predict(X_val)

        fold_metrics = calculate_metrics(y_val, y_pred)
        results.append(fold_metrics)

        print(f"\nFold {fold}/{args.n_splits}")
        print("-" * 35)

        for metric, value in fold_metrics.items():
            print(f"{metric.capitalize():<15}: {value:.4f}")

    results_df = pd.DataFrame(results)
    filename = f"{args.model_name}_cross_eval_results_all.csv"
    path = get_tables_path(filename)
    results_df.to_csv(path)

    print("\nCross-Validation Summary")
    print("=" * 65)

    metrics_table = {"Metric": [], "Mean": [], "STD": []}

    for metric in results_df.columns:
        mean = results_df[metric].mean()
        std = results_df[metric].std(ddof=1)

        metrics_table["Metric"].append(metric.capitalize())
        metrics_table["Mean"].append(mean)
        metrics_table["STD"].append(std)

        print(f"{metric.capitalize():<15}: {mean:.4f} ± {std:.4f}")

    print("=" * 65)

    df = pd.DataFrame(metrics_table)
    path = get_tables_path(f"{args.model_name}_cross_eval_metrics.csv")
    df.to_csv(path)


if __name__ == "__main__":
    main()
