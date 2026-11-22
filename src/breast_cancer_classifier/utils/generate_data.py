from pathlib import Path


def generate_data():
    """Create the project's output directories if they don't exist."""
    project_root = Path.cwd()

    dirs = [
        "data",
        "data/raw",
        "data/interinm",
        "data/processed",
    ]

    for directory in dirs:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
