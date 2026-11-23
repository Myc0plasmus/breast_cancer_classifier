from pathlib import Path

from .generate_artifacts import generate_artifacts


def get_figure_path(filename: str):
    generate_artifacts()
    return Path.cwd() / "artifacts" / "figures" / filename


def get_tables_path(filename: str):
    generate_artifacts()
    return Path.cwd() / "artifacts" / "tables" / filename


def get_rf_models_path(filename: str):
    generate_artifacts()
    return Path.cwd() / "artifacts" / "models" / "random_forest" / filename
