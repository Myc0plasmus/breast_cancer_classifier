import re

from .utils import get_source

MARKER_PATTERN = re.compile(r"{{([a-zA-Z_][\w.]*)}}")


def insert_code_snippets(template: str) -> str:
    def replace(match: re.Match[str]) -> str:
        reference = match.group(1)
        return get_source(reference)

    return MARKER_PATTERN.sub(replace, template)
