import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from breast_cancer_classifier.utils import get_rf_models_path


class RandomForestWrapped:
    def __init__(self, model_name: str, model_path_override: Path | None = None):
        self.model = None
        self.model_name = model_name

        self.try_load_model(model_path_override)

    def try_load_model(self, model_path_override: Path | None = None) -> bool:
        model_path = get_rf_models_path(self.model_name)
        if model_path_override is not None:
            self.model = joblib.load(model_path_override)
            return True
        elif model_path.exists():
            self.model = joblib.load(model_path)
            return True
        return False

    def save_model(self, model_path_override=None):
        if self.model is None:
            raise ValueError("Cannot save: no model has been trained or loaded.")

        if model_path_override is not None:
            joblib.dump(self.model, model_path_override)
        else:
            model_path = get_rf_models_path(self.model_name)
            joblib.dump(self.model, model_path)

    def train(self, X_train, y_train, n_estimators=100, overwrite_model=False):
        if self.model is not None and not overwrite_model:
            raise ValueError(
                "Model has already been loaded or trained, to overwrite the me model the overwrite_model argument needs to be explicitly set to True."
            )

        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=25, random_state=42, n_jobs=-1
        )

        # Train
        model.fit(X_train, y_train)
        self.model = model

    def eval(self, X_test):
        if self.model is None:
            raise ValueError(
                "Model has to be either loaded or trained before evaluation"
            )
        # Predict
        y_pred = self.model.predict(X_test)
        return y_pred
