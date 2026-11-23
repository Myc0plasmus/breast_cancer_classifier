import argparse

from breast_cancer_classifier.utils import DatasetLoader
from breast_cancer_classifier.models import RandomForestWrapped


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier.")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name used to identify and save the model.",
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
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100).",
    )

    parser.add_argument(
        "--overwrite-model",
        action="store_false",
        help="Allow overwriting an already loaded or trained model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load training dataset
    loader = DatasetLoader(dataset_type=args.dataset_type)
    train_dataset = loader.load_train()

    if args.target_column not in train_dataset.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found in training dataset."
        )

    # Separate features and target
    X_train = train_dataset.drop(columns=[args.target_column])
    y_train = train_dataset[args.target_column]

    if X_train.empty or len(y_train) == 0:
        raise ValueError("Training dataset is empty.")

    # Initialize wrapper
    rf = RandomForestWrapped(model_name=args.model_name)

    # Train model
    rf.train(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        overwrite_model=args.overwrite_model,
    )

    # Save model
    rf.save_model()

    print(f"Model '{args.model_name}' trained successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of estimators: {args.n_estimators}")


if __name__ == "__main__":
    main()
