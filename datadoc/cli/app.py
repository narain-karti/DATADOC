import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import os

from datadoc.core.engine import DATADOC

app = typer.Typer(
    help="DATADOC: The Open Source Operating System for Dataset Engineering.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

VERSION = "0.1.0"

BANNER = r"""
 ____    _  _____  _    ____   ___   ____
|  _ \  / \|_   _|/ \  |  _ \ / _ \ / ___|
| | | |/ _ \ | | / _ \ | | | | | | | |
| |_| / ___ \| |/ ___ \| |_| | |_| | |___
|____/_/   \_\_/_/   \_\____/ \___/ \____|
"""


def print_banner():
    banner_text = Text(BANNER, style="bold cyan")
    panel = Panel(
        banner_text,
        subtitle=f"[dim]v{VERSION} - The Open Source OS for Dataset Engineering[/dim]",
        border_style="bright_blue",
        padding=(0, 2),
    )
    console.print(panel)


def print_step(icon: str, message: str, style: str = "bold white"):
    console.print(f"  {icon}  [{style}]{message}[/]")


def load_dataset(file_path: str) -> DATADOC:
    if not os.path.exists(file_path):
        console.print(f"\n  [bold red][X] File not found:[/] {file_path}")
        raise typer.Exit(code=1)

    print_step("[>>]", f"Loading [cyan]{file_path}[/cyan]...")
    try:
        doc = DATADOC(file_path)
    except Exception as e:
        console.print(f"\n  [bold red][X] Failed to read dataset:[/] {e}")
        raise typer.Exit(code=1)

    rows, cols = doc.df.shape
    print_step("[OK]", f"Loaded [green]{rows:,}[/green] rows x [green]{cols}[/green] columns", "bold green")
    return doc


@app.command()
def analyze(file_path: str):
    """
    Scans the dataset and returns a rich health report.
    """
    print_banner()
    doc = load_dataset(file_path)

    print_step("[..]", "Running health analysis...")
    report = doc.analyze()

    console.print()
    table = Table(
        title="[bold]Health Report[/bold]",
        box=box.ROUNDED,
        title_style="bold cyan",
        header_style="bold bright_white",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Metric", justify="left", style="white", min_width=25)
    table.add_column("Value", justify="center", style="bold", min_width=10)
    table.add_column("Status", justify="center", min_width=10)

    table.add_row("Rows", f"{report['rows']:,}", "[green]--[/green]")
    table.add_row("Columns", str(report["cols"]), "[green]--[/green]")

    for p_name, p_stats in report["plugins"].items():
        if p_name == "MissingValuePlugin":
            total = p_stats["total_missing"]
            status = "[green][OK] Clean[/green]" if total == 0 else f"[red][!!] {total} found[/red]"
            table.add_row("Missing Values", str(total), status)
        elif p_name == "OutlierPlugin":
            count = len(p_stats["outlier_columns"])
            status = "[green][OK] Clean[/green]" if count == 0 else f"[yellow][!!] {count} cols[/yellow]"
            table.add_row("Outlier Columns", str(count), status)
        elif p_name == "CategoricalEncoderPlugin":
            count = len(p_stats["categorical_columns"])
            cols_list = ", ".join(p_stats["categorical_columns"]) if count > 0 else "--"
            status = "[dim]None[/dim]" if count == 0 else f"[cyan]{cols_list}[/cyan]"
            table.add_row("Categorical Columns", str(count), status)

    console.print(table)
    console.print()


@app.command()
def recommend(file_path: str):
    """
    Outputs a list of suggested engineering steps without applying them.
    """
    print_banner()
    doc = load_dataset(file_path)

    print_step("[..]", "Generating recommendations...")
    recommendations = doc.recommend()

    console.print()
    if not recommendations:
        console.print(Panel(
            "[bold green][OK] Dataset looks perfectly healthy! No recommendations.[/bold green]",
            border_style="green",
        ))
        return

    rec_table = Table(
        title="[bold]Recommended Actions[/bold]",
        box=box.ROUNDED,
        title_style="bold yellow",
        header_style="bold bright_white",
        border_style="yellow",
        show_lines=True,
    )
    rec_table.add_column("#", justify="center", style="bold yellow", width=4)
    rec_table.add_column("Recommendation", style="white")

    for i, rec in enumerate(recommendations, 1):
        rec_table.add_row(str(i), rec)

    console.print(rec_table)
    console.print()
    console.print("  [dim]Run [bold cyan]datadoc engineer <file>[/bold cyan] to apply these automatically.[/dim]\n")


@app.command()
def engineer(file_path: str):
    """
    Automatically applies best-practice pipelines.
    """
    print_banner()
    doc = load_dataset(file_path)

    print_step("[..]", "Running Rule Engine...")

    # Show which plugins are being applied
    for plugin in doc.plugins:
        analysis = plugin.analyze(doc.df)
        has_work = any(bool(v) for k, v in analysis.items() if k.startswith('has_'))
        if has_work:
            print_step("[>>]", f"Applying {plugin.name}...", "dim")
        else:
            print_step("[--]", f"Skipping {plugin.name} (not needed)", "dim")

    clean_df = doc.engineer()

    output_path = f"clean_{os.path.basename(file_path)}"
    clean_df.to_csv(output_path, index=False)

    console.print()
    console.print(Panel(
        f"[bold green][OK] Success![/bold green]\n\n"
        f"  Input:  [cyan]{file_path}[/cyan] ({doc.df.shape[0]} rows x {doc.df.shape[1]} cols)\n"
        f"  Output: [cyan]{output_path}[/cyan] ({clean_df.shape[0]} rows x {clean_df.shape[1]} cols)",
        title="[bold]Engineering Complete[/bold]",
        border_style="green",
    ))
    console.print()


@app.command()
def pipeline(file_path: str):
    """
    Exports the generated pipeline as a standalone .py script.
    """
    print_banner()
    doc = load_dataset(file_path)

    print_step("[..]", "Generating Python pipeline...")

    script = doc.pipeline()
    output_path = f"pipeline_{os.path.basename(file_path).split('.')[0]}.py"

    with open(output_path, "w") as f:
        f.write(script)

    console.print()
    console.print(Panel(
        f"[bold green][OK] Pipeline generated![/bold green]\n\n"
        f"  Saved to: [bold cyan]{output_path}[/bold cyan]\n\n"
        f"  [dim]Run it with:[/dim] [bold white]python {output_path}[/bold white]",
        title="[bold]Pipeline Export[/bold]",
        border_style="cyan",
    ))
    console.print()


@app.command()
def version():
    """
    Displays the current DATADOC version.
    """
    print_banner()


if __name__ == "__main__":
    app()
