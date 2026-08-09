import typer
from rich.console import Console
import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables (including .env) at startup
load_dotenv()

from datadoc.core.pipeline import (  # noqa: E402
    DataDocError,
    DataDocPipeline,
    PipelineConfig,
    read_dataset,
    write_dataset,
)

app = typer.Typer(
    help="DATADOC: The Open Source Operating System for Dataset Engineering.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _project_version() -> str:
    """Read the source version when running from a checkout, then use metadata."""
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.exists():
        for line in project_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                return line.split('"', 2)[1]
    try:
        from importlib.metadata import version

        return version("datadoc-cli")
    except Exception:
        return "0.4.0"


VERSION = _project_version()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _pipeline_config(
    target: Optional[str] = None,
    task: str = "auto",
    drop_identifiers: bool = False,
    scaling: str = "auto",
    clip_outliers: bool = False,
    estimator_family: str = "linear",
    time_column: Optional[str] = None,
    group_column: Optional[str] = None,
) -> PipelineConfig:
    return PipelineConfig(
        target=target,
        task=task,
        drop_identifiers=drop_identifiers,
        scaling=scaling,
        clip_outliers=clip_outliers,
        estimator_family=estimator_family,
        time_column=time_column,
        group_column=group_column,
    )


def _write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# COMMAND: version
# ──────────────────────────────────────────────────────────────
@app.command()
def version():
    """Displays the current DATADOC version."""
    console.print(f"datadoc-cli {VERSION}")


# ──────────────────────────────────────────────────────────────
# COMMAND: profile
# ──────────────────────────────────────────────────────────────
@app.command()
def profile(
    file_path: str,
    target: Optional[str] = typer.Option(
        None, "--target", help="Target column; it is never transformed."
    ),
    output: Optional[str] = typer.Option(None, "--output", help="Optional JSON report path."),
):
    """Profile a CSV or Parquet dataset without modifying it."""
    try:
        result = (
            DataDocPipeline(_pipeline_config(target=target))
            .profile(read_dataset(file_path))
            .to_dict()
        )
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print_json(json.dumps(result))
    if output:
        _write_json(output, result)
        console.print(f"[green]Saved profile to {output}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: plan
# ──────────────────────────────────────────────────────────────
@app.command(name="plan")
def pipeline_plan(
    file_path: str,
    target: Optional[str] = typer.Option(None, "--target"),
    task: str = typer.Option("auto", "--task", case_sensitive=False),
    drop_identifiers: bool = typer.Option(False, "--drop-identifiers"),
    output: Optional[str] = typer.Option(None, "--output"),
):
    """Create an explainable transformation plan without applying it."""
    config = _pipeline_config(target=target, task=task, drop_identifiers=drop_identifiers)
    try:
        result = DataDocPipeline(config).plan(read_dataset(file_path)).to_dict()
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print_json(json.dumps(result))
    if output:
        _write_json(output, result)


# ──────────────────────────────────────────────────────────────
# COMMAND: fit
# ──────────────────────────────────────────────────────────────
@app.command()
def fit(
    file_path: str,
    output: str = typer.Option("pipeline.json", "--output", help="Output JSON artifact."),
    target: Optional[str] = typer.Option(None, "--target"),
    task: str = typer.Option("auto", "--task", case_sensitive=False),
    drop_identifiers: bool = typer.Option(False, "--drop-identifiers"),
    scaling: str = typer.Option("auto", "--scaling", case_sensitive=False),
    clip_outliers: bool = typer.Option(False, "--clip-outliers"),
):
    """Fit a pipeline only on a training dataset and save its artifact."""
    config = _pipeline_config(target, task, drop_identifiers, scaling, clip_outliers)
    try:
        pipeline = DataDocPipeline(config).fit(read_dataset(file_path))
        pipeline.save(output)
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"[green]Fitted leakage-safe pipeline saved to {output}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: transform
# ──────────────────────────────────────────────────────────────
@app.command()
def transform(
    file_path: str,
    pipeline: str = typer.Option(..., "--pipeline", help="Fitted pipeline JSON artifact."),
    output: str = typer.Option(..., "--output", help="CSV or Parquet output path."),
):
    """Apply a fitted pipeline to validation, test, or inference data."""
    try:
        transformed = DataDocPipeline.load(pipeline).transform(read_dataset(file_path))
        write_dataset(transformed, output)
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"[green]Wrote {transformed.height:,} transformed rows to {output}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: evaluate
# ──────────────────────────────────────────────────────────────
@app.command()
def evaluate(
    file_path: str,
    target: str = typer.Option(..., "--target"),
    task: str = typer.Option("auto", "--task", case_sensitive=False),
    estimator: str = typer.Option("linear", "--estimator", case_sensitive=False),
    time_column: Optional[str] = typer.Option(
        None, "--time-column", help="Use ordered validation based on this column."
    ),
    group_column: Optional[str] = typer.Option(
        None, "--group-column", help="Keep groups separated across splits."
    ),
    output: Optional[str] = typer.Option(None, "--output", help="Optional evaluation JSON report."),
):
    """Benchmark a safe candidate pipeline against a minimal baseline."""
    config = _pipeline_config(
        target=target,
        task=task,
        estimator_family=estimator,
        time_column=time_column,
        group_column=group_column,
    )
    try:
        report = DataDocPipeline(config).evaluate(read_dataset(file_path)).to_dict()
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print_json(json.dumps(report))
    if output:
        _write_json(output, report)


# ──────────────────────────────────────────────────────────────
# COMMAND: export
# ──────────────────────────────────────────────────────────────
@app.command(name="export")
def export_pipeline(
    pipeline: str = typer.Option(..., "--pipeline", help="Fitted pipeline JSON artifact."),
    output: str = typer.Option("pipeline.py", "--output", help="Python export path."),
):
    """Export a small executable wrapper around a fitted pipeline artifact."""
    try:
        exported = DataDocPipeline.load(pipeline).export_python(pipeline)
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    Path(output).write_text(exported, encoding="utf-8")
    console.print(f"[green]Saved executable pipeline wrapper to {output}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: run
# ──────────────────────────────────────────────────────────────
@app.command(name="run")
def run_pipeline(
    file_path: str,
    target: Optional[str] = typer.Option(None, "--target"),
    task: str = typer.Option("auto", "--task", case_sensitive=False),
    output_dir: str = typer.Option("datadoc-run", "--output-dir"),
    evaluate_model: bool = typer.Option(
        False, "--evaluate", help="Run optional scikit-learn benchmark when a target is supplied."
    ),
):
    """Profile, plan, fit, transform, and persist one reproducible local run."""
    directory = Path(output_dir)
    df = read_dataset(file_path)
    config = _pipeline_config(target=target, task=task)
    pl = DataDocPipeline(config)
    profile_result = pl.profile(df).to_dict()
    plan_result = pl.plan(df).to_dict()
    pl.fit(df)
    transformed = pl.transform(df)
    _write_json(directory / "profile.json", profile_result)
    _write_json(directory / "plan.json", plan_result)
    pl.save(directory / "pipeline.json")
    write_dataset(transformed, directory / "transformed.parquet")
    manifest = {
        "input_path": str(Path(file_path).resolve()),
        "schema_fingerprint": profile_result["schema_fingerprint"],
        "target": target,
        "rows": df.height,
        "output_schema": pl.output_schema_,
        "artifact": "pipeline.json",
    }
    if evaluate_model:
        if not target:
            raise typer.BadParameter("--evaluate requires --target.")
        eval_report = DataDocPipeline(config).evaluate(df).to_dict()
        _write_json(directory / "evaluation.json", eval_report)
        baseline = eval_report["baseline_score"]
        candidate = eval_report["selected_score"]
        imp = eval_report["improvement"]
        console.print(
            f"[bold cyan]Evaluation Benchmark ({eval_report['metric']}):[/bold cyan] "
            f"Baseline: [yellow]{baseline:.4f}[/yellow] | "
            f"Candidate: [green]{candidate:.4f}[/green] "
            f"([bold green]+{imp:.4f}[/bold green])"
        )
    console.print(f"[green]Created reproducible DATADOC run in {directory}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: ui
# ──────────────────────────────────────────────────────────────
@app.command()
def ui(
    file_path: str, port: int = typer.Option(8000, "--port", help="Port to run the UI server on.")
):
    """Launch the interactive Web Dashboard."""
    import uvicorn
    from datadoc.cli.ui_server import init_server
    import webbrowser

    console.print(f"[bold cyan]Initializing Dashboard on port {port}...[/bold cyan]")

    init_server(file_path)

    url = f"http://127.0.0.1:{port}"
    console.print(f"\n[bold green]Dashboard is live! Opening browser to: {url}[/bold green]\n")

    webbrowser.open(url)

    uvicorn.run("datadoc.cli.ui_server:app", host="127.0.0.1", port=port, log_level="info")
