import re
from pathlib import Path

from .utils import csv_to_markdown

TABLE_MARKER_PATTERN = re.compile(r"{{\s*table\s+(.+?)\s*}}")


def insert_tables(template: str) -> str:
    def replace(match: re.Match[str]) -> str:
        csv_path = Path(match.group(1).strip())

        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path

        return csv_to_markdown(csv_path)

    return TABLE_MARKER_PATTERN.sub(replace, template)
