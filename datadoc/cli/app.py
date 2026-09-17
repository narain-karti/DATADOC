import sys
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
import json
from pathlib import Path
from typing import Optional, Any
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
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(legacy_windows=False)
plugins_app = typer.Typer(help="Inspect and manage plugins.")
app.add_typer(plugins_app, name="plugins")

VERSION = None


def _project_version() -> str:
    """Read the source version when running from a checkout, then use metadata."""
    global VERSION
    if VERSION:
        return VERSION
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.exists():
        for line in project_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version ="):
                try:
                    VERSION = line.split('"', 2)[1]
                    return VERSION
                except Exception:
                    pass
    try:
        from importlib.metadata import version

        VERSION = version("datadoc-cli")
        return VERSION
    except Exception:
        from datadoc import __version__ as pkg_version

        VERSION = pkg_version
        return VERSION


VERSION = _project_version()

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "scaling": "none",
        "clip_outliers": False,
        "categorical_threshold": 10,
        "estimator_family": "tree",
        "description": "Fast, no scaling/clipping - great for EDA & tree models.",
    },
    "balanced": {
        "scaling": "standard",
        "clip_outliers": True,
        "categorical_threshold": 20,
        "estimator_family": "linear",
        "description": "Standard scaling + IQR clipping - safe default for linear models.",
    },
    "linear": {
        "scaling": "standard",
        "clip_outliers": False,
        "categorical_threshold": 20,
        "estimator_family": "linear",
        "description": "Standard scaling (linear/Ridge/LogReg), no clipping.",
    },
    "tree": {
        "scaling": "none",
        "clip_outliers": False,
        "categorical_threshold": 20,
        "estimator_family": "tree",
        "description": "No scaling/clipping - optimal for RandomForest/XGBoost/LightGBM.",
    },
    "time": {
        "scaling": "standard",
        "clip_outliers": False,
        "categorical_threshold": 20,
        "estimator_family": "linear",
        "description": "Time-series aware: expects --time-column for ordered splits.",
    },
    "robust": {
        "scaling": "robust",
        "clip_outliers": True,
        "categorical_threshold": 20,
        "estimator_family": "linear",
        "description": "Robust scaling (median/IQR) + clipping for heavy-tailed data.",
    },
}


# ──────────────────────────────────────────────────────────────
# Config file helpers (datadoc.toml + pyproject.toml [tool.datadoc])
# ──────────────────────────────────────────────────────────────
def _strip_inline_comment(line: str) -> str:
    """Cut a `#` comment, ignoring hashes inside single/double quotes."""
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _parse_simple_value(text: str) -> Any:
    """Coerce a single TOML scalar (or inline list) without third-party deps."""
    v = text.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_value(part) for part in inner.split(",")]
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def _parse_simple_toml(text: str) -> dict[str, Any]:
    """Fallback tiny TOML parser for key = value and [tool.datadoc].

    Handles inline `#` comments, quoted strings, booleans, numbers, and
    inline lists — enough for `datadoc.toml` on Python 3.10 without `tomli`.
    """
    data: dict[str, Any] = {}
    current_section = None
    for raw in text.splitlines():
        line = _strip_inline_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        parsed = _parse_simple_value(v)
        if current_section == "tool.datadoc":
            data[k] = parsed
        elif current_section is None or current_section == "datadoc":
            data[k] = parsed
    return data


def _load_toml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        # py311+: tomllib
        try:
            import tomllib  # type: ignore

            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except ImportError:
            import tomli as tomllib  # type: ignore

            with open(path, "rb") as f:
                raw = tomllib.load(f)
        if path.name == "pyproject.toml":
            return raw.get("tool", {}).get("datadoc", {})
        return raw.get("datadoc", raw) if isinstance(raw, dict) else {}
    except Exception:
        # fallback simple parser
        try:
            return _parse_simple_toml(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _find_and_load_config(explicit: Optional[str] = None) -> tuple[dict[str, Any], Optional[Path]]:
    if explicit:
        p = Path(explicit)
        if p.exists() and p.suffix == ".json":
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                return raw.get("config", raw), p
            except Exception:
                pass
        return _load_toml_file(p), p if p.exists() else None
    # search upward from cwd
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        for name in ("datadoc.toml", "datadoc.cfg", ".datadoc.toml"):
            cand = parent / name
            if cand.exists():
                return _load_toml_file(cand), cand
        py = parent / "pyproject.toml"
        if py.exists():
            data = _load_toml_file(py)
            if data:
                return data, py
    return {}, None


def _resolve_preset(preset: Optional[str], config: dict[str, Any]) -> dict[str, Any]:
    if not preset:
        return config
    p = PRESETS.get(preset.lower())
    if not p:
        raise typer.BadParameter(f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}")
    # preset provides defaults, config overrides
    merged = {k: v for k, v in p.items() if k != "description"}
    merged.update(config)
    # preset flag itself
    merged["_preset"] = preset
    return merged


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _pipeline_config(
    target: Optional[str] = None,
    task: str = "auto",
    drop_identifiers: bool = False,
    deduplicate: bool = False,
    scaling: str = "auto",
    clip_outliers: bool = False,
    estimator_family: str = "linear",
    time_column: Optional[str] = None,
    group_column: Optional[str] = None,
    categorical_threshold: int = 20,
    rare_category_min_frequency: float = 0.0,
    datetime_cyclical: bool = False,
    no_hour: bool = False,
    identifier_columns: Optional[list] = None,
    ignored_columns: Optional[list] = None,
    strict_schema: bool = True,
    preset: Optional[str] = None,
    config_file: Optional[str] = None,
    config_dict: Optional[dict] = None,
) -> PipelineConfig:
    # load config file if not passed
    file_cfg, _ = _find_and_load_config(config_file) if config_dict is None else (config_dict, None)
    # preset applied first, then file, then explicit args take precedence if not None/default
    if preset:
        preset_vals = {k: v for k, v in PRESETS[preset.lower()].items() if k != "description"}
        for k, v in preset_vals.items():
            if k not in file_cfg:
                file_cfg[k] = v

    # file_cfg provides defaults; explicit args override when they differ from defaults
    # For simplicity, explicit args always override if caller passed non-default
    # We detect by checking if file_cfg has the key and arg is default - keep file value
    # But we already merged preset, now apply explicit
    # Only override file_cfg if explicit != default sentinel
    # Since Typer gives us actual values, we treat them as overrides
    def choose(key: str, explicit: Any, default: Any):
        if explicit != default and explicit is not None:
            return explicit
        return file_cfg.get(key, explicit if explicit is not None else default)

    # Actually simpler: if file_cfg has key and explicit is default, use file
    # We need to know defaults; caller passes defaults, so if file_cfg has entry, use it unless explicit differs
    # Use explicit if it's not equal to default value
    target = choose("target", target, None)
    task = choose("task", task, "auto")
    drop_identifiers = choose("drop_identifiers", drop_identifiers, False)
    deduplicate = choose("deduplicate", deduplicate, False)
    scaling = choose("scaling", scaling, "auto")
    clip_outliers = choose("clip_outliers", clip_outliers, False)
    estimator_family = choose("estimator_family", estimator_family, "linear")
    time_column = choose("time_column", time_column, None)
    group_column = choose("group_column", group_column, None)
    categorical_threshold = choose("categorical_threshold", categorical_threshold, 20)
    rare_category_min_frequency = choose(
        "rare_category_min_frequency", rare_category_min_frequency, 0.0
    )
    datetime_cyclical = choose("datetime_cyclical", datetime_cyclical, False)
    # no_hour inverts datetime_extract_hour
    datetime_extract_hour = not choose("no_hour", no_hour, False)
    if "datetime_extract_hour" in file_cfg and not no_hour:
        datetime_extract_hour = bool(file_cfg.get("datetime_extract_hour", True))
    # repeatable list flags: explicit non-empty wins, else config file, else empty
    if identifier_columns:
        resolved_identifiers = list(identifier_columns)
    else:
        resolved_identifiers = list(file_cfg.get("identifier_columns", []) or [])
    if ignored_columns:
        resolved_ignored = list(ignored_columns)
    else:
        resolved_ignored = list(file_cfg.get("ignored_columns", []) or [])
    strict_schema = choose("strict_schema", strict_schema, True)

    custom_interactions = file_cfg.get("custom_interactions", []) or []
    custom_transforms = file_cfg.get("custom_transforms", []) or []

    return PipelineConfig(
        target=target,
        task=task,  # type: ignore
        drop_identifiers=drop_identifiers,
        deduplicate=deduplicate,
        scaling=scaling,  # type: ignore
        clip_outliers=clip_outliers,
        estimator_family=estimator_family,  # type: ignore
        time_column=time_column,
        group_column=group_column,
        categorical_threshold=categorical_threshold,
        rare_category_min_frequency=float(rare_category_min_frequency),
        datetime_cyclical=datetime_cyclical,
        datetime_extract_hour=datetime_extract_hour,
        identifier_columns=resolved_identifiers,
        ignored_columns=resolved_ignored,
        strict_schema=strict_schema,
        custom_interactions=custom_interactions,
        custom_transforms=custom_transforms,
    )


def _write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_config_used(cfg: PipelineConfig, source: Optional[Path]) -> None:
    if source:
        console.print(f"[dim]Config from {source} merged.[/dim]")
    # no verbose otherwise


# ──────────────────────────────────────────────────────────────
# COMMAND: version
# ──────────────────────────────────────────────────────────────
@app.command()
def version():
    """Displays the current DATADOC version."""
    console.print(f"datadoc-cli {VERSION}")
    console.print("[dim]Presets:[/dim] " + ", ".join(f"{k}" for k in PRESETS))


# ──────────────────────────────────────────────────────────────
# COMMAND: init  (create datadoc.toml)
# ──────────────────────────────────────────────────────────────
@app.command()
def init(
    output: str = typer.Option("datadoc.toml", "--output", "-o", help="Where to write config."),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Preset to prefill: {', '.join(PRESETS)}"
    ),
):
    """Create a starter datadoc.toml config file."""
    cfg = PRESETS.get(preset.lower(), PRESETS["balanced"]) if preset else PRESETS["balanced"]
    content = f"""# DATADOC config - committed for reproducibility
# Docs: https://github.com/narain-karti/DATADOC
target = ""        # e.g. "churn"
task = "auto"      # auto | classification | regression
preset = "{preset or "balanced"}"
drop_identifiers = false
deduplicate = false
scaling = "{cfg.get("scaling", "auto")}"
clip_outliers = {"true" if cfg.get("clip_outliers") else "false"}
categorical_threshold = {cfg.get("categorical_threshold", 20)}
rare_category_min_frequency = 0.0  # e.g. 0.02 groups rare into __RARE__
datetime_cyclical = false
datetime_extract_hour = true
estimator_family = "{cfg.get("estimator_family", "linear")}"
# time_column = ""
# group_column = ""
strict_schema = true
"""
    Path(output).write_text(content, encoding="utf-8")
    console.print(f"[green]Created {output} with preset '{preset or 'balanced'}'[/green]")
    console.print(Panel(content, title="datadoc.toml", border_style="cyan"))


# ──────────────────────────────────────────────────────────────
# COMMAND: profile
# ──────────────────────────────────────────────────────────────
@app.command()
def profile(
    file_path: str = typer.Argument(..., help="CSV or Parquet file."),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column; never transformed.", rich_help_panel="Data"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="JSON report path.", rich_help_panel="Output"
    ),
    compare: Optional[str] = typer.Option(
        None, "--compare", help="Compare with another profile.json.", rich_help_panel="Output"
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Human-readable findings.", rich_help_panel="Display"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to datadoc.toml", rich_help_panel="Config"
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Quick preset: {', '.join(PRESETS)}", rich_help_panel="Config"
    ),
):
    """Profile a CSV or Parquet dataset without modifying it."""
    file_cfg, cfg_path = _find_and_load_config(config)
    if preset:
        file_cfg = _resolve_preset(preset, file_cfg)
    # preset/target may be in config
    target = file_cfg.get("target", target) if target is None else target
    try:
        pipe_cfg = _pipeline_config(
            target=target, config_dict=file_cfg, preset=preset, config_file=config
        )
        result = DataDocPipeline(pipe_cfg).profile(read_dataset(file_path)).to_dict()
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error

    if explain:
        tbl = Table(title="Findings", show_header=True, header_style="bold magenta")
        tbl.add_column("Severity")
        tbl.add_column("Code")
        tbl.add_column("Column")
        tbl.add_column("Message")
        for f in result.get("findings", []):
            tbl.add_row(
                str(f.get("severity")),
                str(f.get("code")),
                str(f.get("column")),
                str(f.get("message")),
            )
        console.print(tbl)
        tbl2 = Table(title="Roles", show_header=True)
        tbl2.add_column("Column")
        tbl2.add_column("Role")
        tbl2.add_column("Confidence")
        tbl2.add_column("Rationale")
        for r in result.get("roles", []):
            tbl2.add_row(r["name"], r["role"], str(r["confidence"]), r["rationale"][:60])
        console.print(tbl2)
        console.print(
            f"[dim]Rows: {result['rows']}  Cols: {result['columns']}  Fingerprint: {result['schema_fingerprint']}[/dim]"
        )
    else:
        console.print_json(json.dumps(result))

    if compare:
        try:
            other = _read_json(compare)
            # simple diff
            console.print(Panel(f"Comparing with {compare}", border_style="yellow"))
            if result.get("schema_fingerprint") != other.get("schema_fingerprint"):
                console.print(
                    f"[yellow]Schema fingerprint differs: {result.get('schema_fingerprint')} vs {other.get('schema_fingerprint')}[/yellow]"
                )
            # null counts diff
            diff_rows = []
            for k in set(
                list(result.get("null_counts", {}).keys())
                + list(other.get("null_counts", {}).keys())
            ):
                a = result.get("null_counts", {}).get(k, 0)
                b = other.get("null_counts", {}).get(k, 0)
                if a != b:
                    diff_rows.append(f"{k}: {b} -> {a}")
            if diff_rows:
                console.print("[cyan]Null count changes:[/cyan] " + "; ".join(diff_rows))
            else:
                console.print("[green]No null count changes[/green]")
        except Exception as e:
            console.print(f"[red]Compare failed:[/red] {e}")

    if output:
        _write_json(output, result)
        console.print(f"[green]Saved profile to {output}[/green]")
    if cfg_path:
        _print_config_used(pipe_cfg, cfg_path)


# ──────────────────────────────────────────────────────────────
# COMMAND: health
# ──────────────────────────────────────────────────────────────
@app.command(name="health")
def health(
    file_path: str = typer.Argument(..., help="CSV or Parquet dataset file."),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column; never transformed.", rich_help_panel="Data"
    ),
):
    """Audit data health, calculate 0-100 quality score, and display findings."""
    from datadoc.core.report import _compute_health_score

    try:
        df = read_dataset(file_path)
        pipe_cfg = _pipeline_config(target=target)
        result = DataDocPipeline(pipe_cfg).profile(df).to_dict()
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error

    score, grade = _compute_health_score(df, result)
    color = "green" if score >= 85 else ("yellow" if score >= 70 else "red")

    console.print(
        Panel(
            f"[bold {color}]Health Score: {score}/100[/bold {color}]  [dim]|[/dim]  "
            f"[bold {color}]Grade: {grade}[/bold {color}]  [dim]|[/dim]  "
            f"Rows: [bold]{df.height:,}[/bold]  [dim]|[/dim]  "
            f"Cols: [bold]{df.width}[/bold]  [dim]|[/dim]  "
            f"Nulls: [bold]{sum(result.get('null_counts', {}).values()):,}[/bold]",
            title="DATADOC Health Audit",
            border_style=color,
        )
    )

    findings = result.get("findings", [])
    if findings:
        tbl = Table(title="Quality Findings", show_header=True, header_style="bold magenta")
        tbl.add_column("Severity", style="bold")
        tbl.add_column("Code")
        tbl.add_column("Column")
        tbl.add_column("Message")
        for f in findings:
            sev = str(f.get("severity", "info")).lower()
            sev_color = (
                "red"
                if sev in {"high", "critical", "error"}
                else ("yellow" if sev in {"medium", "warning"} else "cyan")
            )
            tbl.add_row(
                f"[{sev_color}]{f.get('severity')}[/{sev_color}]",
                str(f.get("code", "")),
                str(f.get("column", "")),
                str(f.get("message", "")),
            )
        console.print(tbl)
    else:
        console.print("[green]No quality findings. Dataset is clean![/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: plan
# ──────────────────────────────────────────────────────────────
@app.command(name="plan")
def pipeline_plan(
    file_path: str = typer.Argument(..., help="CSV or Parquet file."),
    target: Optional[str] = typer.Option(None, "--target", "-t", rich_help_panel="Data"),
    task: str = typer.Option("auto", "--task", case_sensitive=False, rich_help_panel="Data"),
    drop_identifiers: bool = typer.Option(
        False, "--drop-identifiers", help="Drop ID-like columns.", rich_help_panel="Data"
    ),
    identifier_column: Optional[list[str]] = typer.Option(
        None,
        "--identifier-column",
        help="Repeatable: force a column to identifier role (e.g. PassengerId).",
        rich_help_panel="Data",
    ),
    ignore_column: Optional[list[str]] = typer.Option(
        None,
        "--ignore-column",
        help="Repeatable: exclude a column from features.",
        rich_help_panel="Data",
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", rich_help_panel="Output"),
    explain_flag: bool = typer.Option(
        False, "--explain", help="Human-readable plan.", rich_help_panel="Display"
    ),
    diff: Optional[str] = typer.Option(
        None, "--diff", help="Diff against previous plan.json", rich_help_panel="Output"
    ),
    config: Optional[str] = typer.Option(None, "--config", rich_help_panel="Config"),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Preset: {', '.join(PRESETS)}", rich_help_panel="Config"
    ),
):
    """Create an explainable transformation plan without applying it."""
    file_cfg, cfg_path = _find_and_load_config(config)
    if preset:
        file_cfg = _resolve_preset(preset, file_cfg)
    # config file provides defaults; _pipeline_config.choose() lets explicit flags win
    pipe_cfg = _pipeline_config(
        target=target,
        task=task,
        drop_identifiers=drop_identifiers,
        identifier_columns=identifier_column,
        ignored_columns=ignore_column,
        config_dict=file_cfg,
        preset=preset,
        config_file=config,
    )  # type: ignore
    try:
        result = DataDocPipeline(pipe_cfg).plan(read_dataset(file_path)).to_dict()
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error

    if explain_flag:
        tbl = Table(title="Planned Operations", show_header=True)
        tbl.add_column("#")
        tbl.add_column("Operation")
        tbl.add_column("Column")
        tbl.add_column("Reason")
        for i, op in enumerate(result.get("operations", []), 1):
            tbl.add_row(
                str(i), op.get("operation", ""), op.get("column", ""), op.get("reason", "")[:70]
            )
        console.print(tbl)
        if result.get("findings"):
            console.print(f"[dim]{len(result['findings'])} findings - see profile --explain[/dim]")
    else:
        console.print_json(json.dumps(result))

    if diff:
        try:
            other = _read_json(diff)
            a_ops = {(o["operation"], o["column"]) for o in result.get("operations", [])}
            b_ops = {(o["operation"], o["column"]) for o in other.get("operations", [])}
            added = a_ops - b_ops
            removed = b_ops - a_ops
            if added:
                console.print(f"[yellow]Only in current plan:[/yellow] {added}")
            if removed:
                console.print(f"[yellow]Only in {diff}:[/yellow] {removed}")
            if not added and not removed:
                console.print("[green]Plans identical[/green]")
        except Exception as e:
            console.print(f"[red]Diff failed:[/red] {e}")

    if output:
        _write_json(output, result)
        console.print(f"[green]Saved plan to {output}[/green]")
    if cfg_path:
        _print_config_used(pipe_cfg, cfg_path)


# ──────────────────────────────────────────────────────────────
# COMMAND: fit
# ──────────────────────────────────────────────────────────────
@app.command()
def fit(
    file_path: str = typer.Argument(..., help="Training CSV/Parquet."),
    output: str = typer.Option(
        "pipeline.json", "--output", "-o", help="Output JSON artifact.", rich_help_panel="Output"
    ),
    target: Optional[str] = typer.Option(None, "--target", "-t", rich_help_panel="Data"),
    task: str = typer.Option("auto", "--task", case_sensitive=False, rich_help_panel="Data"),
    drop_identifiers: bool = typer.Option(False, "--drop-identifiers", rich_help_panel="Data"),
    identifier_column: Optional[list[str]] = typer.Option(
        None,
        "--identifier-column",
        help="Repeatable: force a column to identifier role.",
        rich_help_panel="Data",
    ),
    ignore_column: Optional[list[str]] = typer.Option(
        None,
        "--ignore-column",
        help="Repeatable: exclude a column from features.",
        rich_help_panel="Data",
    ),
    deduplicate: bool = typer.Option(
        False, "--deduplicate", help="Drop duplicate rows at fit.", rich_help_panel="Data"
    ),
    scaling: str = typer.Option("auto", "--scaling", case_sensitive=False, rich_help_panel="Data"),
    clip_outliers: bool = typer.Option(False, "--clip-outliers", rich_help_panel="Data"),
    categorical_threshold: int = typer.Option(
        20, "--categorical-threshold", rich_help_panel="Data"
    ),
    rare_frequency: float = typer.Option(
        0.0,
        "--rare-frequency",
        help="Group cats below this freq into __RARE__.",
        rich_help_panel="Data",
    ),
    cyclical: bool = typer.Option(
        False, "--cyclical", help="Add sin/cos for datetime parts.", rich_help_panel="Data"
    ),
    no_hour: bool = typer.Option(
        False, "--no-hour", help="Skip hour extraction.", rich_help_panel="Data"
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Preset: {', '.join(PRESETS)}", rich_help_panel="Config"
    ),
    config: Optional[str] = typer.Option(None, "--config", rich_help_panel="Config"),
    strict_schema: bool = typer.Option(
        True, "--strict-schema/--no-strict-schema", rich_help_panel="Data"
    ),
):
    """Fit a pipeline only on a training dataset and save its artifact."""
    file_cfg, cfg_path = _find_and_load_config(config)
    if preset:
        file_cfg = _resolve_preset(preset, file_cfg)
    pipe_cfg = _pipeline_config(
        target=target,
        task=task,
        drop_identifiers=drop_identifiers,
        identifier_columns=identifier_column,
        ignored_columns=ignore_column,
        deduplicate=deduplicate,
        scaling=scaling,
        clip_outliers=clip_outliers,
        categorical_threshold=categorical_threshold,
        rare_category_min_frequency=rare_frequency,
        datetime_cyclical=cyclical,
        no_hour=no_hour,
        strict_schema=strict_schema,
        preset=preset,
        config_file=config,
        config_dict=file_cfg,
    )  # type: ignore
    try:
        pipeline = DataDocPipeline(pipe_cfg).fit(read_dataset(file_path))
        pipeline.save(output)
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"[green]Fitted leakage-safe pipeline saved to {output}[/green]")
    console.print(
        f"[dim]Input schema: {len(pipeline.input_schema_)} cols -> Output: {len(pipeline.output_schema_)} cols[/dim]"
    )
    if cfg_path:
        _print_config_used(pipe_cfg, cfg_path)


# ──────────────────────────────────────────────────────────────
# COMMAND: transform
# ──────────────────────────────────────────────────────────────
@app.command()
def transform(
    file_path: str = typer.Argument(..., help="CSV/Parquet to transform."),
    pipeline: str = typer.Option(
        ..., "--pipeline", "-p", help="Fitted pipeline JSON artifact.", rich_help_panel="Pipeline"
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="CSV or Parquet output path.", rich_help_panel="Output"
    ),
    validate: bool = typer.Option(
        False, "--validate", help="Run schema & drift PSI checks.", rich_help_panel="Checks"
    ),
):
    """Apply a fitted pipeline to validation, test, or inference data."""
    try:
        pipe = DataDocPipeline.load(pipeline)
        df = read_dataset(file_path)
        if validate:
            report = pipe.drift_report(df)
            if report["schema_ok"]:
                console.print("[green]Schema validation: OK[/green]")
            for issue in report["issues"]:
                if issue.get("type") == "schema":
                    console.print(f"[yellow]Schema warning:[/yellow] {issue['message']}")
                else:
                    console.print(f"[yellow]Drift:[/yellow] {issue['message']}")
            if not report["issues"]:
                console.print("[green]No drift detected[/green]")
        transformed = pipe.transform(df)
        write_dataset(transformed, output)
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"[green]Wrote {transformed.height:,} transformed rows to {output}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: evaluate
# ──────────────────────────────────────────────────────────────
@app.command()
def evaluate(
    file_path: str = typer.Argument(..., help="CSV/Parquet to evaluate."),
    target: str = typer.Option(..., "--target", "-t", rich_help_panel="Data"),
    task: str = typer.Option("auto", "--task", case_sensitive=False, rich_help_panel="Data"),
    estimator: str = typer.Option(
        "linear", "--estimator", case_sensitive=False, rich_help_panel="Model"
    ),
    time_column: Optional[str] = typer.Option(
        None, "--time-column", help="Ordered temporal split.", rich_help_panel="Split"
    ),
    group_column: Optional[str] = typer.Option(
        None, "--group-column", help="Keep groups separated.", rich_help_panel="Split"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", rich_help_panel="Output"),
    ablation: bool = typer.Option(
        False, "--ablation", help="Per-component ablation report.", rich_help_panel="Output"
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Preset: {', '.join(PRESETS)}", rich_help_panel="Config"
    ),
    config: Optional[str] = typer.Option(None, "--config", rich_help_panel="Config"),
):
    """Benchmark a safe candidate pipeline against a minimal baseline."""
    file_cfg, cfg_path = _find_and_load_config(config)
    if preset:
        file_cfg = _resolve_preset(preset, file_cfg)
    pipe_cfg = _pipeline_config(
        target=target,
        task=task,  # type: ignore
        estimator_family=estimator,  # type: ignore
        time_column=time_column,
        group_column=group_column,
        preset=preset,
        config_file=config,
        config_dict=file_cfg,
    )
    try:
        pipe = DataDocPipeline(pipe_cfg)
        report = pipe.evaluate(read_dataset(file_path)).to_dict()
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print_json(json.dumps(report))
    if ablation:
        try:
            abl = DataDocPipeline(pipe_cfg).evaluate_ablation(read_dataset(file_path))
            console.print(Panel(f"Ablation (metric={abl['metric']})", border_style="cyan"))
            console.print_json(json.dumps(abl))
            if output:
                _write_json(str(Path(output).with_suffix("")) + ".ablation.json", abl)
        except Exception as e:
            console.print(f"[red]Ablation failed:[/red] {e}")
    if output:
        _write_json(output, report)
        console.print(f"[green]Saved evaluation to {output}[/green]")
    if cfg_path:
        _print_config_used(pipe_cfg, cfg_path)


# ──────────────────────────────────────────────────────────────
# COMMAND: export
# ──────────────────────────────────────────────────────────────
@app.command(name="export")
def export_pipeline(
    pipeline: str = typer.Option(
        ..., "--pipeline", "-p", help="Fitted pipeline JSON artifact.", rich_help_panel="Pipeline"
    ),
    output: str = typer.Option(
        "pipeline.py", "--output", "-o", help="Python export path.", rich_help_panel="Output"
    ),
    format: str = typer.Option(
        "python",
        "--format",
        "-f",
        case_sensitive=False,
        help="python | sklearn | joblib",
        rich_help_panel="Output",
    ),
):
    """Export an executable wrapper around a fitted pipeline artifact."""
    try:
        pipe = DataDocPipeline.load(pipeline)
        if format.lower() in ("python", "py"):
            exported = pipe.export_python(pipeline)
            Path(output).write_text(exported, encoding="utf-8")
        elif format.lower() in ("sklearn", "joblib"):
            try:
                saved = pipe.export_sklearn_artifact(output)
                console.print(
                    f"[green]Saved sklearn/joblib artifact to {saved} (joblib.load -> dict)[/green]"
                )
                console.print(
                    "[dim]Loader: import joblib; art=joblib.load(path); then DataDocPipeline.load(path.json) or rebuild from dict[/dim]"
                )
                return
            except DataDocError as e:
                raise typer.BadParameter(str(e))
            except Exception as e:
                raise typer.BadParameter(str(e))
        else:
            raise typer.BadParameter("Unknown format. Use python | sklearn | joblib")
    except DataDocError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"[green]Saved executable pipeline wrapper to {output}[/green]")


# ──────────────────────────────────────────────────────────────
# COMMAND: run
# ──────────────────────────────────────────────────────────────
@app.command(name="run")
def run_pipeline(
    file_path: str = typer.Argument(..., help="CSV/Parquet to run on."),
    target: Optional[str] = typer.Option(None, "--target", "-t", rich_help_panel="Data"),
    task: str = typer.Option("auto", "--task", case_sensitive=False, rich_help_panel="Data"),
    output_dir: str = typer.Option("datadoc-run", "--output-dir", rich_help_panel="Output"),
    evaluate_model: bool = typer.Option(
        False, "--evaluate", help="Run benchmark when target supplied.", rich_help_panel="Eval"
    ),
    ablation: bool = typer.Option(
        False, "--ablation", help="Also write evaluation ablation.", rich_help_panel="Eval"
    ),
    # parity flags from fit
    drop_identifiers: bool = typer.Option(False, "--drop-identifiers", rich_help_panel="Data"),
    identifier_column: Optional[list[str]] = typer.Option(
        None, "--identifier-column", rich_help_panel="Data"
    ),
    ignore_column: Optional[list[str]] = typer.Option(
        None, "--ignore-column", rich_help_panel="Data"
    ),
    deduplicate: bool = typer.Option(False, "--deduplicate", rich_help_panel="Data"),
    scaling: str = typer.Option("auto", "--scaling", case_sensitive=False, rich_help_panel="Data"),
    clip_outliers: bool = typer.Option(False, "--clip-outliers", rich_help_panel="Data"),
    categorical_threshold: int = typer.Option(
        20, "--categorical-threshold", rich_help_panel="Data"
    ),
    rare_frequency: float = typer.Option(0.0, "--rare-frequency", rich_help_panel="Data"),
    cyclical: bool = typer.Option(False, "--cyclical", rich_help_panel="Data"),
    no_hour: bool = typer.Option(False, "--no-hour", rich_help_panel="Data"),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Preset: {', '.join(PRESETS)}", rich_help_panel="Config"
    ),
    config: Optional[str] = typer.Option(None, "--config", rich_help_panel="Config"),
    strict_schema: bool = typer.Option(
        True, "--strict-schema/--no-strict-schema", rich_help_panel="Data"
    ),
):
    """Profile, plan, fit, transform, and persist one reproducible local run."""
    file_cfg, cfg_path = _find_and_load_config(config)
    if preset:
        file_cfg = _resolve_preset(preset, file_cfg)
    pipe_cfg = _pipeline_config(
        target=target,
        task=task,  # type: ignore
        drop_identifiers=drop_identifiers,
        identifier_columns=identifier_column,
        ignored_columns=ignore_column,
        deduplicate=deduplicate,
        scaling=scaling,  # type: ignore
        clip_outliers=clip_outliers,
        categorical_threshold=categorical_threshold,
        rare_category_min_frequency=rare_frequency,
        datetime_cyclical=cyclical,
        no_hour=no_hour,
        strict_schema=strict_schema,
        preset=preset,
        config_file=config,
        config_dict=file_cfg,
    )
    directory = Path(output_dir)
    df = read_dataset(file_path)
    pl = DataDocPipeline(pipe_cfg)
    # profile + plan first, then fit reuses cached profile_/plan_
    profile_result = pl.profile(df).to_dict()
    plan_result = pl.plan(df).to_dict()
    pl.fit(df)
    transformed = pl.transform(df)
    _write_json(directory / "profile.json", profile_result)
    _write_json(directory / "plan.json", plan_result)
    pl.save(directory / "pipeline.json")
    write_dataset(transformed, directory / "transformed.parquet")
    _write_json(
        directory / "manifest.json",
        {
            "input_path": str(Path(file_path).resolve()),
            "schema_fingerprint": profile_result["schema_fingerprint"],
            "target": pipe_cfg.target,
            "rows": df.height,
            "output_schema": pl.output_schema_,
            "artifact": "pipeline.json",
            "config": pipe_cfg.__dict__,
            "provenance": getattr(pl, "train_provenance_", {}),
            "datadoc_version": VERSION,
            "output_dir": str(directory.resolve()),
        },
    )
    console.print(f"[green]Profile->Plan->Fit->Transform done -> {directory}[/green]")
    if evaluate_model:
        if not pipe_cfg.target:
            raise typer.BadParameter("--evaluate requires --target (via flag or datadoc.toml).")
        eval_report = DataDocPipeline(pipe_cfg).evaluate(df).to_dict()
        _write_json(directory / "evaluation.json", eval_report)
        if ablation:
            abl = DataDocPipeline(pipe_cfg).evaluate_ablation(df)
            _write_json(directory / "evaluation.ablation.json", abl)
        baseline = eval_report["baseline_score"]
        candidate = eval_report["selected_score"]
        imp = eval_report["improvement"]
        console.print(
            f"[bold cyan]Evaluation Benchmark ({eval_report['metric']}):[/bold cyan] "
            f"Baseline: [yellow]{baseline:.4f}[/yellow] | "
            f"Candidate: [green]{candidate:.4f}[/green] "
            f"([bold green]{imp:+.4f}[/bold green])"
        )
        console.print(
            f"[dim]Selected: {eval_report['selected_pipeline']}  Features: {eval_report['feature_count']}[/dim]"
        )
    console.print(f"[bold green]Created reproducible DATADOC run in {directory}[/bold green]")
    if cfg_path:
        _print_config_used(pipe_cfg, cfg_path)


# ──────────────────────────────────────────────────────────────
# COMMAND: wizard
# ──────────────────────────────────────────────────────────────
@app.command()
def wizard(
    file_path: str = typer.Argument(..., help="CSV/Parquet to guide."),
    output_dir: str = typer.Option("datadoc-run", "--output-dir", help="Run dir."),
):
    """Interactive wizard: asks target, preset, and runs profile->plan->fit."""
    df = read_dataset(file_path)
    console.print(
        Panel(
            f"DATADOC Wizard - {file_path} ({df.height} rows, {df.width} cols)", border_style="cyan"
        )
    )
    # suggest columns
    cols = df.columns
    console.print(f"Columns: [cyan]{', '.join(cols[:12])}{'...' if len(cols) > 12 else ''}[/cyan]")
    target = Prompt.ask("Target column (blank = unsupervised)", default="", show_default=False)
    target = target.strip() or None
    if target and target not in cols:
        console.print(f"[red]Unknown column {target}, continuing without target.[/red]")
        target = None
    preset = Prompt.ask(f"Preset {list(PRESETS.keys())}", default="balanced")
    preset = preset.lower().strip() if preset else "balanced"
    if preset not in PRESETS:
        preset = "balanced"
    console.print(f"[dim]{PRESETS[preset]['description']}[/dim]")
    drop = Confirm.ask("Drop suspected identifiers (customer_id, uuid)?", default=False)
    dedup = Confirm.ask("Drop duplicate rows at fit?", default=False)
    # build config
    file_cfg: dict[str, Any] = {}
    file_cfg = _resolve_preset(preset, file_cfg)
    file_cfg["drop_identifiers"] = drop
    file_cfg["deduplicate"] = dedup
    if target:
        file_cfg["target"] = target
    # ask scaling/clip override
    scaling = Prompt.ask(
        "Scaling auto|none|standard|robust", default=file_cfg.get("scaling", "auto")
    )
    clip = Confirm.ask("Enable IQR clipping?", default=file_cfg.get("clip_outliers", False))
    rare = Prompt.ask("Rare frequency (0=off, e.g. 0.02)", default="0.0")
    try:
        rare_f = float(rare)
    except Exception:
        rare_f = 0.0
    file_cfg["scaling"] = scaling
    file_cfg["clip_outliers"] = clip
    file_cfg["rare_category_min_frequency"] = rare_f

    # write datadoc.toml
    toml_path = Path("datadoc.toml")
    if not toml_path.exists():
        init(output=str(toml_path), preset=preset)
        # patch target etc.
        txt = toml_path.read_text(encoding="utf-8")
        if target:
            txt = txt.replace('target = ""', f'target = "{target}"')
        txt = txt.replace(
            "drop_identifiers = false", f"drop_identifiers = {'true' if drop else 'false'}"
        )
        txt = txt.replace(f'scaling = "{PRESETS["balanced"]["scaling"]}"', f'scaling = "{scaling}"')
        # naive patch for preset line
        toml_path.write_text(txt, encoding="utf-8")
        console.print(f"[green]Wrote {toml_path}[/green]")

    # run (typer.Option defaults are placeholders when calling directly,
    # so every parameter must be passed explicitly here)
    run_pipeline(
        file_path,
        target=target,
        task="auto",
        output_dir=output_dir,
        evaluate_model=bool(target),
        ablation=False,
        drop_identifiers=drop,
        identifier_column=None,
        ignore_column=None,
        deduplicate=dedup,
        scaling=scaling,
        clip_outliers=clip,
        categorical_threshold=20,
        rare_frequency=rare_f,
        cyclical=False,
        no_hour=False,
        preset=preset,
        config=None,
        strict_schema=True,
    )


# ──────────────────────────────────────────────────────────────
# COMMAND: diff
# ──────────────────────────────────────────────────────────────
@app.command(name="diff")
def diff_cmd(
    file_a: str = typer.Argument(..., help="First JSON (profile/plan/pipeline)"),
    file_b: str = typer.Argument(..., help="Second JSON"),
):
    """Diff two DATADOC JSON artifacts (profile, plan, or pipeline)."""
    a = _read_json(file_a)
    b = _read_json(file_b)
    # generic top-level keys diff
    tbl = Table(title=f"Diff: {Path(file_a).name} <-> {Path(file_b).name}")
    tbl.add_column("Key")
    tbl.add_column(Path(file_a).name[:20])
    tbl.add_column(Path(file_b).name[:20])
    tbl.add_column("Status")
    keys = sorted(set(a.keys()) | set(b.keys()))
    diffs = 0
    for k in keys:
        av = json.dumps(a.get(k), sort_keys=True)[:80]
        bv = json.dumps(b.get(k), sort_keys=True)[:80]
        status = "same" if a.get(k) == b.get(k) else "diff"
        if status == "diff":
            diffs += 1
            tbl.add_row(k, av, bv, "[red]diff[/red]")
        else:
            tbl.add_row(k, av, bv, "[green]same[/green]")
    console.print(tbl)
    if diffs == 0:
        console.print("[green]Files identical (top-level).[/green]")
    else:
        console.print(f"[yellow]{diffs} top-level keys differ[/yellow]")

    # plan operations diff if present
    if "operations" in a and "operations" in b:
        a_ops = {(o.get("operation"), o.get("column")) for o in a["operations"]}
        b_ops = {(o.get("operation"), o.get("column")) for o in b["operations"]}
        added = a_ops - b_ops
        removed = b_ops - a_ops
        if added:
            console.print(f"[yellow]Only in A ({Path(file_a).name}):[/yellow] {added}")
        if removed:
            console.print(f"[yellow]Only in B ({Path(file_b).name}):[/yellow] {removed}")


# ──────────────────────────────────────────────────────────────
# COMMAND: explain
# ──────────────────────────────────────────────────────────────
@app.command(name="explain")
def explain(
    file_path: str = typer.Argument(..., help="CSV or Parquet dataset file."),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column for predictive task.", rich_help_panel="Data"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model (e.g., gpt-4o-mini, gemini/gemini-2.0-flash, ollama/llama3).",
        rich_help_panel="AI",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save AI explanation to markdown file.",
        rich_help_panel="Output",
    ),
    recommend_config: bool = typer.Option(
        False,
        "--recommend-config",
        help="Write recommended datadoc.toml based on AI analysis.",
        rich_help_panel="Config",
    ),
):
    """AI-powered dataset feature engineering hypotheses, missingness analysis, and data quality audit."""
    from rich.markdown import Markdown
    from datadoc.ai.explainer import explain_dataset

    try:
        df = read_dataset(file_path)
    except Exception as e:
        raise typer.BadParameter(f"Could not load dataset: {e}")

    with console.status(
        "[bold cyan]Synthesizing feature engineering hypotheses & data quality audit..."
    ):
        try:
            result = explain_dataset(df, target=target, model=model)
        except Exception as e:
            console.print(f"[bold red]AI Analysis Failed:[/bold red] {e}")
            raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]DATADOC AI Dataset Explainer[/bold cyan]\n"
            f"Provider: [bold]{result.provider}[/bold]  [dim]|[/dim]  "
            f"Model: [bold]{result.model}[/bold]  [dim]|[/dim]  "
            f"Recommended Preset: [bold green]{result.recommended_preset}[/bold green]",
            border_style="cyan",
        )
    )

    console.print(Markdown(result.raw_markdown))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.raw_markdown, encoding="utf-8")
        console.print(f"\n[bold green]Saved AI explanation to:[/bold green] {out_path.resolve()}")

    if recommend_config:
        cfg_content = f"""# Recommended datadoc.toml generated by DATADOC AI
task = "{result.recommended_config.get("task", "auto")}"
preset = "{result.recommended_preset}"
drop_identifiers = true
deduplicate = false
clip_outliers = {str(result.recommended_config.get("clip_outliers", True)).lower()}
categorical_threshold = 20
rare_category_min_frequency = 0.01
strict_schema = true
"""
        target_cfg = Path("datadoc.toml")
        target_cfg.write_text(cfg_content, encoding="utf-8")
        console.print(
            f"[bold green]Saved recommended configuration to {target_cfg.resolve()}[/bold green]"
        )
        console.print(Panel(cfg_content, title="datadoc.toml", border_style="green"))


# ──────────────────────────────────────────────────────────────
# COMMAND: report
# ──────────────────────────────────────────────────────────────
@app.command()
def report(
    file_path: str = typer.Argument(..., help="CSV or Parquet file to analyze."),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column name.", rich_help_panel="Data"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output HTML report file path.", rich_help_panel="Output"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", help="Custom report title.", rich_help_panel="Display"
    ),
    pipeline: Optional[str] = typer.Option(
        None,
        "--pipeline",
        "-p",
        help="Path to fitted pipeline.json to include lineage.",
        rich_help_panel="Pipeline",
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Quick preset: {', '.join(PRESETS)}", rich_help_panel="Config"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to datadoc.toml", rich_help_panel="Config"
    ),
    ai: bool = typer.Option(
        False,
        "--ai",
        help="Include AI Executive Summary & Feature Engineering Hypotheses in report.",
        rich_help_panel="AI",
    ),
    ai_model: Optional[str] = typer.Option(
        None,
        "--ai-model",
        help="Model to use for AI report analysis (e.g. gpt-4o-mini, gemini/gemini-2.0-flash).",
        rich_help_panel="AI",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open report in browser automatically.",
        rich_help_panel="Display",
    ),
):
    """Generate an automated, standalone HTML report for sharing."""
    import webbrowser
    from datadoc.core.report import generate_html_report

    file_cfg, _ = _find_and_load_config(config)
    if preset:
        file_cfg = _resolve_preset(preset, file_cfg)
    target = file_cfg.get("target", target) if target is None else target

    try:
        df = read_dataset(file_path)
    except Exception as e:
        raise typer.BadParameter(f"Could not load dataset: {e}")

    pipe_instance = None
    if pipeline:
        try:
            pipe_instance = DataDocPipeline.load(pipeline)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load pipeline from {pipeline}: {e}[/yellow]")

    ai_explanation = None
    if ai:
        with console.status("[bold cyan]Analyzing dataset with AI for report..."):
            from datadoc.ai.explainer import explain_dataset

            res = explain_dataset(df, target=target, model=ai_model)
            ai_explanation = res.raw_markdown

    html_content = generate_html_report(
        df=df,
        target=target,
        title=title,
        pipeline=pipe_instance,
        dataset_name=Path(file_path).name,
        ai_explanation=ai_explanation,
    )

    out_path = (
        Path(output)
        if output
        else Path(file_path).with_suffix("").with_name(f"{Path(file_path).stem}_report.html")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")

    console.print(f"[bold green]Created standalone report:[/bold green] {out_path.resolve()}")
    if open_browser:
        try:
            webbrowser.open(out_path.resolve().as_uri())
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# COMMAND: compare
# ──────────────────────────────────────────────────────────────
@app.command()
def compare(
    raw_path: str = typer.Argument(..., help="Path to raw dataset (CSV or Parquet)."),
    transformed_path: str = typer.Argument(
        ..., help="Path to transformed dataset (CSV or Parquet)."
    ),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column name.", rich_help_panel="Data"
    ),
    html: Optional[str] = typer.Option(
        None,
        "--html",
        help="Path to save interactive HTML comparison report.",
        rich_help_panel="Output",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output comparison metrics as JSON.", rich_help_panel="Output"
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open HTML report in browser automatically.",
        rich_help_panel="Display",
    ),
):
    """Visually compare raw vs. transformed datasets side-by-side."""
    import webbrowser
    from datadoc.core.compare import compare_datasets, print_comparison_table, compare_to_html

    try:
        raw_df = read_dataset(raw_path)
        trans_df = read_dataset(transformed_path)
    except Exception as e:
        raise typer.BadParameter(f"Error reading datasets: {e}")

    comp = compare_datasets(raw_df, trans_df, target=target)

    if json_output:
        console.print_json(json.dumps(comp.to_dict(), indent=2))
    else:
        print_comparison_table(comp, console=console)

    if html:
        html_content = compare_to_html(
            comp, title=f"Comparison: {Path(raw_path).name} vs {Path(transformed_path).name}"
        )
        out_path = Path(html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding="utf-8")
        console.print(
            f"[bold green]Saved HTML comparison report:[/bold green] {out_path.resolve()}"
        )
        if open_browser:
            try:
                webbrowser.open(out_path.resolve().as_uri())
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────
# COMMAND: lint
# ──────────────────────────────────────────────────────────────
@app.command(name="lint")
def lint_cmd(
    file_path: str = typer.Argument(..., help="CSV/Parquet to lint for leakage risks."),
    target: Optional[str] = typer.Option(None, "--target", "-t", rich_help_panel="Data"),
    config: Optional[str] = typer.Option(None, "--config", rich_help_panel="Config"),
):
    """Lint a dataset for leakage & prep pitfalls."""
    file_cfg, _ = _find_and_load_config(config)
    target = file_cfg.get("target", target) if target is None else target
    df = read_dataset(file_path)
    findings = []
    # duplicate target check
    if target and target in df.columns:
        for c in df.columns:
            if c != target and df[c].equals(df[target]):
                findings.append(
                    f"[red]LEAKAGE:[/red] Column '{c}' exactly duplicates target '{target}'"
                )
        if df[target].null_count() > 0:
            findings.append(
                f"[yellow]Target '{target}' has {df[target].null_count()} nulls - resolve labels before fit[/yellow]"
            )
    # infinite
    for name in df.columns:
        s = df[name]
        if s.dtype.is_float() and s.is_infinite().any():
            findings.append(f"[yellow]Infinite values in '{name}' will be treated as null[/yellow]")
    # high cardinality identifier risk
    for name in df.columns:
        if name != target and df[name].dtype.is_numeric():
            non_null = df[name].drop_nulls()
            if non_null.len() > 0 and non_null.n_unique() / non_null.len() > 0.9:
                findings.append(
                    f"[yellow]Column '{name}' has >90% unique numeric values — likely an identifier[/yellow]"
                )
    # duplicate rows
    dup = int(df.is_duplicated().sum()) if df.height else 0
    if dup:
        findings.append(
            f"[yellow]{dup} duplicate rows detected - consider deduplication (not auto-dropped)[/yellow]"
        )
    # constant
    for name in df.columns:
        if df[name].drop_nulls().n_unique() <= 1:
            findings.append(f"[dim]Constant column '{name}' will be dropped[/dim]")
    # profile findings
    pipe_cfg = _pipeline_config(target=target, config_dict=file_cfg, config_file=config)  # type: ignore
    profile = DataDocPipeline(pipe_cfg).profile(df)
    for f in profile.findings:
        findings.append(
            f"{f['severity'].upper()}: {f['code']} - {f['message']} (col={f['column']})"
        )

    if not findings:
        console.print("[green]Lint: no issues - leakage-safe to proceed[/green]")
    else:
        console.print(Panel("\n".join(findings), title="Lint report", border_style="yellow"))
        # guidance
        console.print("[dim]Fix with: datadoc fit --drop-identifiers --preset balanced[/dim]")


# ──────────────────────────────────────────────────────────────
# PLUGINS subcommands
# ──────────────────────────────────────────────────────────────
@plugins_app.command("list")
def plugins_list():
    """List all built-in and entry-point plugins."""
    try:
        from datadoc.plugins.registry import list_plugins

        plugs = list_plugins()
    except Exception:
        # fallback to builtins
        from datadoc.plugins.missing_values import MissingValuePlugin
        from datadoc.plugins.outliers import OutlierPlugin
        from datadoc.plugins.datetime_feat import DatetimePlugin
        from datadoc.plugins.encoders import CategoricalEncoderPlugin
        from datadoc.plugins.scaling import ScalingPlugin

        plugs = [
            MissingValuePlugin(),
            OutlierPlugin(),
            DatetimePlugin(),
            CategoricalEncoderPlugin(),
            ScalingPlugin(),
        ]
        plugs = sorted(plugs, key=lambda p: p.priority)

    tbl = Table(title="Registered Plugins", show_header=True, header_style="bold cyan")
    tbl.add_column("Priority")
    tbl.add_column("Name")
    tbl.add_column("Version")
    tbl.add_column("Description")
    for p in plugs:
        tbl.add_row(str(p.priority), p.name, p.version, p.description)
    console.print(tbl)
    console.print(
        f"[dim]{len(plugs)} plugins. Core 5 + registry extensions. See CONTRIBUTING.md[/dim]"
    )


@plugins_app.command("show")
def plugins_show(name: str = typer.Argument(..., help="Plugin name (e.g. MissingValuePlugin)")):
    """Show details for one plugin."""
    try:
        from datadoc.plugins.registry import get_plugin

        p = get_plugin(name)
        if not p:
            raise typer.BadParameter(f"Unknown plugin '{name}'. Try datadoc plugins list")
    except Exception as e:
        raise typer.BadParameter(str(e))
    console.print(
        Panel(
            f"{p.name} v{p.version}\nPriority {p.priority}\n{p.description}\nDeps: {p.dependencies}\n\n{p.explain()}",
            title=p.name,
            border_style="cyan",
        )
    )


# ──────────────────────────────────────────────────────────────
# COMMAND: ui
# ──────────────────────────────────────────────────────────────
@app.command()
def ui(
    file_path: str = typer.Argument(..., help="File to load in dashboard."),
    port: int = typer.Option(
        8000, "--port", help="Port to run the UI server on.", rich_help_panel="Server"
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open browser.", rich_help_panel="Server"
    ),
):
    """Launch the interactive Web Dashboard."""
    import uvicorn
    from datadoc.cli.ui_server import init_server
    import webbrowser

    console.print(f"[bold cyan]Initializing Dashboard on port {port}...[/bold cyan]")
    init_server(file_path)
    url = f"http://127.0.0.1:{port}"
    console.print(f"\n[bold green]Dashboard is live![/bold green]  {url}\n")
    console.print("[dim]Press Ctrl+C to stop. Dashboard reads from local session 'local'.[/dim]")
    if not no_browser:
        webbrowser.open(url)
    uvicorn.run("datadoc.cli.ui_server:app", host="127.0.0.1", port=port, log_level="info")


# ──────────────────────────────────────────────────────────────
# COMMAND: agent
# ──────────────────────────────────────────────────────────────
@app.command()
def agent(
    file_path: str = typer.Argument(..., help="CSV or Parquet dataset file."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column. If omitted, agent will prompt or infer.", rich_help_panel="Data"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Number of auto-research loops.", rich_help_panel="Agent"),
    interactive: bool = typer.Option(False, "--interactive", help="Prompt user for domain knowledge.", rich_help_panel="Agent"),
    model: Optional[str] = typer.Option(None, "--model", help="Preferred LLM to use.", rich_help_panel="Agent"),
    output: str = typer.Option("pipeline.json", "--output", "-o", help="Output JSON artifact.", rich_help_panel="Output"),
):
    """Run the Agentic Auto-Research Loop to discover features."""
    try:
        from datadoc.ai.agent_runner import DataDocAgent
    except ImportError as e:
        raise typer.BadParameter("pip install 'datadoc-cli[ai]' for agent features.") from e
    
    df = read_dataset(file_path)

    # Resolve / auto-detect target column
    if not target:
        common_candidates = [
            "survived", "target", "label", "churn", "status", "class", "price", "sale_price", "outcome", "y"
        ]
        detected_col = None
        for col in df.columns:
            if col.lower() in common_candidates:
                detected_col = col
                break

        if interactive:
            if detected_col:
                use_detected = typer.confirm(
                    f"Agent detected potential target column '{detected_col}'. Use this?", default=True
                )
                if use_detected:
                    target = detected_col
                else:
                    console.print(f"[dim]Available columns: {', '.join(df.columns)}[/dim]")
                    target = typer.prompt("Please specify target column name")
            else:
                console.print(f"[dim]Available columns: {', '.join(df.columns)}[/dim]")
                target = typer.prompt("Please specify target column name")
        else:
            if detected_col:
                console.print(f"[dim]Auto-detected target column: '{detected_col}'[/dim]")
                target = detected_col
            else:
                console.print(
                    f"[red]Error: Target column must be specified via --target when non-interactive. Available columns: {', '.join(df.columns)}[/red]"
                )
                raise typer.Exit(1)

    # Case-insensitive resolution
    if target not in df.columns:
        match = next((c for c in df.columns if c.lower() == target.lower()), None)
        if match:
            target = match
        else:
            console.print(
                f"[red]Error: Target column '{target}' not found in dataset. Columns: {', '.join(df.columns)}[/red]"
            )
            raise typer.Exit(1)
    
    try:
        agent_runner = DataDocAgent(target=target, model=model)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
        
    final_config = agent_runner.run_interactive(df, iterations=iterations, interactive=interactive)
    
    console.print("\n[cyan]Fitting final pipeline with discovered features...[/cyan]")
    pipeline = DataDocPipeline(final_config).fit(df, target=target)
    pipeline.save(output)
    
    console.print(f"[green]Agent finished. Final leakage-safe pipeline saved to {output}[/green]")
    console.print(f"[dim]Run `datadoc plan {file_path} --config {output} --explain` to see the full plan.[/dim]")


if __name__ == "__main__":
    app()
