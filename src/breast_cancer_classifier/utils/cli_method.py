from typing import Callable
from functools import wraps

from rich.console import Console

from typing import Any, overload


@overload
def cli_method[**P, R](
    func: Callable[P, R],
    message: str,
) -> Callable[P, R]: ...


@overload
def cli_method[**P, R](
    *,
    message: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def cli_method(
    func: Callable[..., Any] | None = None,
    message: str = "",
) -> Any:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            console = Console()
            with console.status(f"[bold green]{message}..."):
                result = f(*args, **kwargs)
            console.print(f"{message}: [green]completed")
            return result

        return wrapper

    if func is None:
        return decorator

    return decorator(func)
