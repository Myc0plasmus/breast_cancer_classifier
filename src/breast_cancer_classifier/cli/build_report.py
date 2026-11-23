from pathlib import Path
from rich.console import Console
import pypandoc

from breast_cancer_classifier.presentation import insert_code_snippets, insert_tables


def build_report() -> None:
    template_path = Path.cwd() / "docs" / "templates" / "report-template.md"
    output_path = Path.cwd() / "docs" / "report-final.md"
    pdf_path = Path.cwd() / "docs" / "report-final.pdf"

    console = Console()

    with console.status("[bold green]Acquiring template..."):
        template = template_path.read_text(encoding="utf-8")
    console.print("Acquiring template: [green]completed")

    with console.status("[bold green]Inserting code snippets..."):
        result = insert_code_snippets(template)
    console.print("Inseting code snippets: [green]completed")

    with console.status("[bold green]Inserting tables..."):
        result = insert_tables(result)
    console.print("Inseting tables: [green]completed")

    with console.status("[bold green]Writing final report..."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
    console.print("Writing final report: [green]completed")

    with console.status("[bold green]Generating PDF..."):
        pypandoc.convert_file(
            str(output_path),
            to="pdf",
            outputfile=str(pdf_path),
            extra_args=[
                "--standalone",
                "--pdf-engine=typst",
            ],
        )
    console.print("Generating PDF: [green]completed")


if __name__ == "__main__":
    build_report()
