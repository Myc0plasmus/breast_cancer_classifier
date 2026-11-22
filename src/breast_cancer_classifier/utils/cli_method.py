from typing import Callable
from functools import wraps

from rich.console import Console


def cli_method[**P, R](func: Callable[P, R], message: str) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        console = Console()
        with console.status(f"[bold green]{message}..."):
            result = func(*args, **kwargs)
        console.print(f"{message}: [green]completed")
        return result

    return wrapper
