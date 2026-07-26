import typer
from rich.console import Console
from rich.table import Table
import os

from datadoc.core.engine import DATADOC

app = typer.Typer(help="DATADOC: The Open Source Operating System for Dataset Engineering.")
console = Console()

@app.command()
def analyze(file_path: str):
    """
    Scans the dataset and returns a rich health report table.
    """
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/] File '{file_path}' not found.")
        raise typer.Exit(code=1)
        
    console.print(f"[bold green]Analyzing {file_path}...[/]")
    
    try:
        doc = DATADOC(file_path)
        report = doc.analyze()
    except Exception as e:
        console.print(f"[bold red]Failed to read dataset:[/] {e}")
        raise typer.Exit(code=1)

    table = Table(title=f"Dataset Health Report: {file_path}")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Rows", str(report["rows"]))
    table.add_row("Columns", str(report["cols"]))
    
    # Iterate plugin reports
    for p_name, p_stats in report["plugins"].items():
        if p_name == "MissingValuePlugin":
            table.add_row("Total Missing Values", str(p_stats["total_missing"]))
    
    console.print(table)
    
@app.command()
def recommend(file_path: str):
    """
    Outputs a list of suggested engineering steps without applying them.
    """
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/] File '{file_path}' not found.")
        raise typer.Exit(code=1)
        
    console.print(f"[bold green]Generating recommendations for {file_path}...[/]")
    
    try:
        doc = DATADOC(file_path)
        recommendations = doc.recommend()
    except Exception as e:
        console.print(f"[bold red]Failed to process dataset:[/] {e}")
        raise typer.Exit(code=1)

    if not recommendations:
        console.print("[bold cyan]Dataset looks perfectly healthy! No recommendations.[/]")
        return
        
    console.print("\n[bold underline]Recommended Actions:[/]")
    for i, rec in enumerate(recommendations, 1):
        console.print(f"[bold yellow]{i}.[/] {rec}")
    console.print()

@app.command()
def engineer(file_path: str):
    """
    Automatically applies best-practice pipelines.
    """
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/] File '{file_path}' not found.")
        raise typer.Exit(code=1)
        
    console.print(f"[bold yellow]Engineering features for {file_path}...[/]")
    
    doc = DATADOC(file_path)
    clean_df = doc.engineer()
    
    output_path = f"clean_{os.path.basename(file_path)}"
    clean_df.to_csv(output_path, index=False)
    
    console.print(f"[bold green]Success![/] Clean dataset saved to [bold cyan]{output_path}[/]")

if __name__ == "__main__":
    app()
