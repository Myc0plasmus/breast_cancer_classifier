from pathlib import Path
from rich.console import Console
import pickle as pkl

from .process_template import process_template
from random_graph_generation.utils import get_graphs_path, get_figure_path, get_tables_path

generation_methods = [
        "erdos_renyi"
        ]

def main() -> None:
    template_path = Path.cwd() / "docs" / "report-template.md"
    output_path = Path.cwd() / "docs" / "report-final.md"

    console = Console()
    for method in generation_methods:

        with console.status(f"[bold green]Loading {method} graphs..."):
            graphs_filename = f"{method}_graphs.pkl"
            graphs_path = get_graphs_path(graphs_filename)
            with open(graphs_path, "rb") as f:
                graphs = pkl.load(f)

        console.print(f"Loading {method} graphs: [green]completed")

    with console.status("[bold green]Acquiring template..."):
        template = template_path.read_text(encoding="utf-8")
    console.print(f"Acquiring template: [green]completed")

    with console.status("[bold green]Inserting code snippets..."):
        result = process_template(template)
    console.print(f"Inseting code snippeks: [green]completed")

    with console.status("[bold green]Writing final report..."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
    console.print(f"Writing final report: [green]completed")


if __name__ == "__main__":
    main()
