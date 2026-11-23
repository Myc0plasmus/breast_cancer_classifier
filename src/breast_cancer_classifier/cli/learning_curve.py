import argparse

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from breast_cancer_classifier.utils import DatasetLoader, get_figure_path
from breast_cancer_classifier.models import RandomForestWrapped


METRICS = ["accuracy", "precision", "recall", "f1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Random Forest learning curve."
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
        "--n-splits", type=int, default=5, help="Number of cross-validation folds."
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest.",
    )

    parser.add_argument(
        "--train-sizes",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Training set fractions to evaluate.",
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


def get_stratified_subset(X, y, fraction, random_state):
    """Select a stratified subset of the training data."""

    if fraction >= 1.0:
        return X, y

    selected_indices = []

    for label in y.unique():
        class_indices = y.index[y == label]

        n_samples = max(1, int(round(len(class_indices) * fraction)))

        n_samples = min(n_samples, len(class_indices))

        sampled = pd.Series(class_indices).sample(
            n=n_samples, random_state=random_state
        )

        selected_indices.extend(sampled.tolist())

    return X.loc[selected_indices], y.loc[selected_indices]


def main():
    args = parse_args()

    # Validate training sizes
    if any(size <= 0 or size > 1 for size in args.train_sizes):
        raise ValueError("Training sizes must be greater than 0 and at most 1.")

    if len(set(args.train_sizes)) != len(args.train_sizes):
        raise ValueError("Training sizes must be unique.")

    train_sizes = sorted(args.train_sizes)

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

    if y.value_counts().min() < args.n_splits:
        raise ValueError(f"Each class must have at least {args.n_splits} samples.")

    # Initialize cross-validation
    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.random_state
    )

    # Initialize wrapper
    rf = RandomForestWrapped(model_name=args.model_name)

    results = []

    print("\nGenerating Learning Curve")
    print("=" * 65)

    for size in train_sizes:
        print(f"\nTraining size: {size:.0%}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]

            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Select stratified subset
            X_train, y_train = get_stratified_subset(
                X_fold_train, y_fold_train, size, args.random_state + fold
            )

            # Train fresh model
            rf.train(
                X_train, y_train, n_estimators=args.n_estimators, overwrite_model=True
            )

            # Calculate training metrics
            train_predictions = rf.predict(X_train)
            train_metrics = calculate_metrics(y_train, train_predictions)

            # Calculate validation metrics
            val_predictions = rf.predict(X_val)
            val_metrics = calculate_metrics(y_val, val_predictions)

            results.append(
                {
                    "train_size": size,
                    "fold": fold,
                    "n_samples": len(X_train),
                    **{
                        f"train_{metric}": value
                        for metric, value in train_metrics.items()
                    },
                    **{f"val_{metric}": value for metric, value in val_metrics.items()},
                }
            )

            print(f"  Fold {fold}/{args.n_splits} completed ({len(X_train)} samples)")

    # Aggregate results
    results_df = pd.DataFrame(results)

    summary = results_df.groupby("train_size").agg(
        {
            "n_samples": "mean",
            **{
                f"{prefix}_{metric}": ["mean", "std"]
                for prefix in ["train", "val"]
                for metric in METRICS
            },
        }
    )

    # Create chart
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    x = summary["n_samples"]["mean"]

    for ax, metric in zip(axes, METRICS):
        train_mean = summary[f"train_{metric}"]["mean"]
        train_std = summary[f"train_{metric}"]["std"].fillna(0)

        val_mean = summary[f"val_{metric}"]["mean"]
        val_std = summary[f"val_{metric}"]["std"].fillna(0)

        ax.plot(x, train_mean, marker="o", label="Training score")

        ax.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.15)

        ax.plot(x, val_mean, marker="o", label="Validation score")

        ax.fill_between(x, val_mean - val_std, val_mean + val_std, alpha=0.15)

        ax.set_title(metric.capitalize())
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Random Forest Learning Curves", fontsize=15)

    fig.tight_layout()

    filename = f"{args.model_name}_learning_curve.png"
    path = get_figure_path(filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")

    print(f"\nLearning curve saved to: {path}")

    plt.show()


if __name__ == "__main__":
    main()
