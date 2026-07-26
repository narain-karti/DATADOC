import typer
from rich.console import Console
from rich.table import Table
import pandas as pd
import os

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
        df = pd.read_csv(file_path)
    except Exception as e:
        console.print(f"[bold red]Failed to read dataset:[/] {e}")
        raise typer.Exit(code=1)

    # Basic analysis
    rows, cols = df.shape
    missing_total = df.isnull().sum().sum()
    
    table = Table(title=f"Dataset Health Report: {file_path}")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Rows", str(rows))
    table.add_row("Columns", str(cols))
    table.add_row("Total Missing Values", str(missing_total))
    
    console.print(table)
    
@app.command()
def engineer(file_path: str):
    """
    Automatically applies best-practice pipelines.
    """
    console.print(f"[bold yellow]Engineering features for {file_path} (Coming Soon!)[/]")

if __name__ == "__main__":
    app()
