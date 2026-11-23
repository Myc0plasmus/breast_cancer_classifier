import csv
from pathlib import Path


def csv_to_markdown(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        return ""

    def escape_cell(value: str) -> str:
        return value.replace("|", r"\|").replace("\n", " ")

    rows = [[escape_cell(cell) for cell in row] for row in rows]

    # Normalize row lengths so the Markdown table is rectangular.
    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]

    header = rows[0]
    body = rows[1:]

    separator = ["---"] * column_count

    markdown_rows = [
        "| " + " | ".join(row) + " |" for row in [header, separator, *body]
    ]

    return "\n".join(markdown_rows)
