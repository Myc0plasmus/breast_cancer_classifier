from pathlib import Path

from .generate_artifacts import generate_artifacts

def get_raw_path(filename: str):
    generate_artifacts()
    return Path.cwd() / "data" / "raw" / filename
