import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import os
import time
from dotenv import load_dotenv

# Load environment variables (including .env) at startup
load_dotenv()

# Determine the default model from the environment, fallback to Groq
DEFAULT_MODEL = os.getenv("DATADOC_MODEL", "groq/llama-3.3-70b-versatile")

from datadoc.core.engine import DATADOC  # noqa: E402

app = typer.Typer(
    help="DATADOC: The Open Source Operating System for Dataset Engineering.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
VERSION = "0.2.0"
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
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"\n  [bold red][X] Failed to read dataset:[/] {e}")
        raise typer.Exit(code=1)

    rows, cols = doc.df.height, doc.df.width
    print_step("[OK]", f"Loaded [green]{rows:,}[/green] rows x [green]{cols}[/green] columns", "bold green")
    return doc


def _get_api_key(model: str) -> str:
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "gpt": "OPENAI_API_KEY", "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    key_name = next((v for k, v in key_map.items() if model.startswith(k)), "API_KEY")
    api_key = os.getenv("OPENAI_API_KEY") if key_name == "API_KEY" else os.getenv(key_name)

    if not api_key:
        console.print(f"\n  [bold yellow][!][/bold yellow] {key_name} not found in environment or .env file.")
        api_key = typer.prompt(f"Please enter your {key_name}", hide_input=True)
    return api_key

# ──────────────────────────────────────────────────────────────
# COMMAND: analyze
# ──────────────────────────────────────────────────────────────
@app.command()
def analyze(
    file_path: str,
    ai: bool = typer.Option(False, "--ai", help="Use AI to generate an Executive Summary of the dataset health."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="The LiteLLM model string to use.")
):
    """
    Scans the dataset and returns a rich health report.
    """
    print_banner()
    doc = load_dataset(file_path)

    with console.status("[bold cyan]Running health analysis...", spinner="dots"):
        report = doc.analyze()
    print_step("[OK]", "Analysis complete.", "bold green")

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
    table.add_column("Status", justify="center", min_width=15)

    table.add_row("Rows", f"{report['rows']:,}", "[green]--[/green]")
    table.add_row("Columns", str(report["cols"]), "[green]--[/green]")

    for p_name, p_stats in report["plugins"].items():
        if p_name == "MissingValuePlugin":
            total = p_stats["total_missing"]
            status = "[green][OK] Clean[/green]" if total == 0 else f"[red][!!] {total} found[/red]"
            table.add_row("Missing Values", str(total), status)
        elif p_name == "OutlierPlugin":
            count = len(p_stats.get("outlier_columns", []))
            status = "[green][OK] Clean[/green]" if count == 0 else f"[yellow][!!] {count} cols[/yellow]"
            table.add_row("Outlier Columns", str(count), status)
        elif p_name == "DatetimePlugin":
            count = len(p_stats.get("datetime_columns", []))
            if count > 0:
                cols_list = ", ".join(p_stats["datetime_columns"])
                table.add_row("Datetime Columns", str(count), f"[cyan]{cols_list}[/cyan]")
            else:
                table.add_row("Datetime Columns", "0", "[dim]None[/dim]")
        elif p_name == "CategoricalEncoderPlugin":
            count = len(p_stats.get("categorical_columns", []))
            if count > 0:
                cols_list = ", ".join(p_stats["categorical_columns"])
                table.add_row("Categorical Columns", str(count), f"[cyan]{cols_list}[/cyan]")
            else:
                table.add_row("Categorical Columns", "0", "[dim]None[/dim]")
        elif p_name == "ScalingPlugin":
            has_issue = p_stats.get("has_scale_issues", False)
            ratio = p_stats.get("scale_ratio", 0)
            if has_issue:
                table.add_row("Scale Mismatch", f"{ratio}x", "[yellow][!!] Needs scaling[/yellow]")
            else:
                table.add_row("Scale Mismatch", "--", "[green][OK] Balanced[/green]")

    console.print(table)
    console.print()

    if ai:
        api_key = _get_api_key(model)
        with console.status(f"[bold cyan]Generating AI Executive Summary ({model})...", spinner="dots"):
            summary = doc.ai_analyze(model=model, api_key=api_key)
        
        from rich.markdown import Markdown
        console.print(Panel(
            Markdown(summary),
            title="[bold blue]AI Executive Summary[/bold blue]",
            border_style="blue"
        ))
        console.print()


# ──────────────────────────────────────────────────────────────
# COMMAND: recommend
# ──────────────────────────────────────────────────────────────
@app.command()
def recommend(
    file_path: str,
    ai: bool = typer.Option(False, "--ai", help="Use AI to generate a custom execution plan without applying it."),
    goal: str = typer.Option("Clean the dataset for machine learning", "--goal", help="The semantic goal for the AI Planner."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="The LiteLLM model string to use.")
):
    """
    Outputs a list of suggested engineering steps without applying them.
    """
    print_banner()
    doc = load_dataset(file_path)

    if ai:
        api_key = _get_api_key(model)
        with console.status(f"[bold cyan]Generating AI Recommendations ({model})...", spinner="dots"):
            recs = doc.ai_recommend(model=model, goal=goal, api_key=api_key)
        
        print_step("[OK]", "AI Recommendations ready.", "bold green")
        console.print()
        
        if not recs:
            console.print(Panel(
                "[bold green][OK] AI determined no steps are necessary for this goal.[/bold green]",
                border_style="green",
            ))
            return

        rec_table = Table(
            title=f"[bold]AI Recommended Plan for: {goal}[/bold]",
            box=box.ROUNDED,
            title_style="bold yellow",
            header_style="bold bright_white",
            border_style="yellow",
            show_lines=True,
        )
        rec_table.add_column("#", justify="center", style="bold yellow", width=4)
        rec_table.add_column("Plugin", style="bold cyan")
        rec_table.add_column("AI Reason", style="white")

        from rich.markdown import Markdown
        for i, rec in enumerate(recs, 1):
            rec_table.add_row(str(i), rec["plugin"], Markdown(rec["reason"]))

        console.print(rec_table)
        console.print()
        console.print(f"  [dim]Run [bold cyan]datadoc engineer \"{file_path}\" --ai --goal \"{goal}\"[/bold cyan] to apply this plan.[/dim]\n")
        return

    with console.status("[bold cyan]Generating recommendations...", spinner="dots"):
        recommendations = doc.recommend()
    print_step("[OK]", "Recommendations ready.", "bold green")

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


# ──────────────────────────────────────────────────────────────
# COMMAND: engineer
# ──────────────────────────────────────────────────────────────
@app.command()
def engineer(
    file_path: str,
    ai: bool = typer.Option(False, "--ai", help="Use the AI Planner to dynamically orchestrate the pipeline."),
    goal: str = typer.Option("Clean the dataset for machine learning", "--goal", help="The semantic goal for the AI Planner."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="The LiteLLM model string to use.")
):
    """
    Automatically applies best-practice pipelines or uses AI to plan.
    """
    print_banner()
    doc = load_dataset(file_path)

    console.print()
    if ai:
        api_key = _get_api_key(model)

        with console.status(f"[bold cyan]Running AI Planner ({model})...", spinner="dots") as status:
            def on_progress(plugin_name, p_status, details):
                if p_status == "running":
                    status.update(f"[bold cyan]Running [yellow]{plugin_name}[/yellow]...[/bold cyan]")
                    time.sleep(0.2)
                elif p_status == "applied":
                    print_step("[>>]", f"Applied [bold cyan]{plugin_name}[/bold cyan]", "bold white")
                    for detail in details:
                        console.print(f"      [dim]-> {detail}[/dim]")
                elif p_status == "skipped":
                    print_step("[--]", f"Skipped [dim]{plugin_name}[/dim]", "dim white")
                elif p_status == "error":
                    console.print(f"\n  [bold red][X] Error in AI Planner:[/] {details[0]}")
                    raise typer.Exit(code=1)

            clean_df = doc.ai_engineer(model=model, goal=goal, api_key=api_key, progress_callback=on_progress)
    else:
        with console.status("[bold cyan]Running Rule Engine...", spinner="dots") as status:
            def on_progress(plugin_name, p_status, details):
                if p_status == "running":
                    status.update(f"[bold cyan]Running [yellow]{plugin_name}[/yellow]...[/bold cyan]")
                    time.sleep(0.5)
                elif p_status == "applied":
                    print_step("[>>]", f"Applied [bold cyan]{plugin_name}[/bold cyan]", "bold white")
                    for detail in details:
                        console.print(f"      [dim]-> {detail}[/dim]")
                elif p_status == "skipped":
                    print_step("[--]", f"Skipped [dim]{plugin_name}[/dim] (not needed)", "dim white")
    
            clean_df = doc.engineer(progress_callback=on_progress)

    output_path = f"clean_{os.path.basename(file_path)}"
    clean_df.write_csv(output_path)

    console.print()
    console.print(Panel(
        f"[bold green][OK] Success![/bold green]\n\n"
        f"  Input:  [cyan]{file_path}[/cyan] ({doc.df.height} rows x {doc.df.width} cols)\n"
        f"  Output: [cyan]{output_path}[/cyan] ({clean_df.height} rows x {clean_df.width} cols)\n\n"
        f"  Applied:  {', '.join(doc._applied_plugins) or 'None'}\n"
        f"  Skipped:  {', '.join(doc._skipped_plugins) or 'None'}",
        title="[bold]Engineering Complete[/bold]",
        border_style="green",
    ))
    console.print()


# ──────────────────────────────────────────────────────────────
# COMMAND: compare
# ──────────────────────────────────────────────────────────────
@app.command()
def compare(file_path: str):
    """
    Shows a diff-like comparison between the raw and engineered dataset.
    """
    print_banner()
    doc = load_dataset(file_path)

    with console.status("[bold cyan]Engineering dataset for comparison...", spinner="dots"):
        clean_df = doc.engineer()
    diff = doc.compare(clean_df)

    console.print()
    table = Table(
        title="[bold]Before vs After Comparison[/bold]",
        box=box.ROUNDED,
        title_style="bold magenta",
        header_style="bold bright_white",
        border_style="magenta",
        show_lines=True,
    )
    table.add_column("Metric", style="white", min_width=20)
    table.add_column("Before", justify="center", style="red", min_width=15)
    table.add_column("After", justify="center", style="green", min_width=15)

    orig_r, orig_c = diff["original_shape"]
    clean_r, clean_c = diff["clean_shape"]
    table.add_row("Rows", str(orig_r), str(clean_r))
    table.add_row("Columns", str(orig_c), str(clean_c))
    table.add_row("Missing Values", str(diff["original_missing"]), str(diff["clean_missing"]))
    table.add_row(
        "Columns Added",
        "--",
        f"+{diff['cols_added']}" if diff["cols_added"] > 0 else "0"
    )

    # Data type breakdown
    for dtype, count in diff["original_dtypes"].items():
        dtype_str = str(dtype)
        clean_count = diff["clean_dtypes"].get(dtype, 0)
        table.add_row(f"dtype: {dtype_str}", str(count), str(clean_count))

    # Check for new dtypes in clean that weren't in original
    for dtype, count in diff["clean_dtypes"].items():
        if dtype not in diff["original_dtypes"]:
            table.add_row(f"dtype: {str(dtype)}", "0", str(count))

    console.print(table)

    # Show applied/skipped plugins
    console.print()
    if hasattr(doc, '_applied_plugins'):
        print_step("[OK]", f"Applied: {', '.join(doc._applied_plugins)}", "bold green")
        print_step("[--]", f"Skipped: {', '.join(doc._skipped_plugins)}", "dim")
    console.print()


# ──────────────────────────────────────────────────────────────
# COMMAND: visualize
# ──────────────────────────────────────────────────────────────
@app.command()
def visualize(file_path: str):
    """
    Generates a massive, interactive terminal dashboard comparing the before and after states.
    """
    import plotext as plt
    import sys
    import polars as pl
    
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        
    print_banner()
    doc = load_dataset(file_path)

    with console.status("[bold cyan]Generating terminal dashboard...", spinner="dots"):
        clean_df = doc.engineer()
        
        orig_missing = {c: doc.df[c].null_count() for c in doc.df.columns}
        clean_missing = {c: clean_df[c].null_count() for c in clean_df.columns}
        cols_with_missing = [c for c, count in orig_missing.items() if count > 0]
        
        # Missing values bar chart
        if cols_with_missing:
            plt.clf()
            plt.theme("dark")
            orig_vals = [orig_missing[c] for c in cols_with_missing]
            clean_vals = [clean_missing.get(c, 0) for c in cols_with_missing]
            
            plt.multiple_bar(cols_with_missing, [orig_vals, clean_vals], labels=["Before", "After"])
            plt.title("Missing Values Resolution")
            plt.plotsize(100, 20)
            missing_plot = plt.build()
        else:
            missing_plot = "  [dim]No missing values found in the original dataset.[/dim]"
            
        # Numerical distributions
        num_cols = [c for c in doc.df.columns if doc.df[c].dtype in pl.NUMERIC_DTYPES]
        dist_plots = []
        
        for col in num_cols:
            if col not in clean_df.columns:
                continue
                
            orig_data = doc.df[col].drop_nulls()
            clean_data = clean_df[col].drop_nulls()
            
            if len(orig_data) == 0 or len(clean_data) == 0:
                continue
                
            plt.clf()
            plt.theme("dark")
            
            plt.hist(orig_data.to_list(), bins=20, label="Before", color="red")
            plt.hist(clean_data.to_list(), bins=20, label="After", color="green")
            plt.title(f"Distribution: {col}")
            plt.plotsize(100, 20)
            dist_plots.append(plt.build())

    # Render directly to terminal
    console.print()
    console.print(Panel(
        Text.from_ansi(missing_plot) if "No missing" not in missing_plot else missing_plot,
        title="[bold yellow]Missing Values Breakdown[/bold yellow]",
        border_style="yellow"
    ))
    
    for i, d_plot in enumerate(dist_plots):
        console.print(Panel(
            Text.from_ansi(d_plot),
            title=f"[bold cyan]Numerical Distribution: {num_cols[i]}[/bold cyan]",
            border_style="cyan"
        ))

    console.print(Panel(
        f"[bold green]Terminal dashboard generated successfully![/bold green]\n\n"
        f"  Applied Plugins: {', '.join(doc._applied_plugins) or 'None'}",
        title="[bold]Summary[/bold]",
        border_style="green",
    ))


# ──────────────────────────────────────────────────────────────
# COMMAND: pipeline
# ──────────────────────────────────────────────────────────────
@app.command()
def pipeline(file_path: str):
    """
    Exports the generated pipeline as a standalone .py script.
    """
    print_banner()
    doc = load_dataset(file_path)

    with console.status("[bold cyan]Generating Python pipeline...", spinner="dots"):
        script = doc.pipeline()
    print_step("[OK]", "Pipeline generated.", "bold green")

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


# ──────────────────────────────────────────────────────────────
# COMMAND: report
# ──────────────────────────────────────────────────────────────
@app.command()
def report(file_path: str):
    """
    Generates a Markdown report summarizing the dataset and recommendations.
    """
    print_banner()
    doc = load_dataset(file_path)

    with console.status("[bold cyan]Generating report...", spinner="dots"):
        report_data = doc.analyze()
        recommendations = doc.recommend()
        plugin_info = doc.list_plugins()

    md_lines = [
        f"# DATADOC Report: {os.path.basename(file_path)}",
        "",
        "---",
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| File | `{file_path}` |",
        f"| Rows | {report_data['rows']:,} |",
        f"| Columns | {report_data['cols']} |",
        "",
    ]

    # Plugin analysis details
    md_lines.append("## Health Analysis")
    md_lines.append("")
    for p_name, p_stats in report_data["plugins"].items():
        md_lines.append(f"### {p_name}")
        md_lines.append("")
        for key, val in p_stats.items():
            md_lines.append(f"- **{key}**: `{val}`")
        md_lines.append("")

    # Recommendations
    md_lines.append("## Recommendations")
    md_lines.append("")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            md_lines.append(f"{i}. {rec}")
    else:
        md_lines.append("No issues found. Dataset is healthy!")
    md_lines.append("")

    # Plugin registry
    md_lines.append("## Plugin Registry")
    md_lines.append("")
    md_lines.append("| Plugin | Version | Priority | Will Trigger |")
    md_lines.append("|--------|---------|----------|-------------|")
    for p in plugin_info:
        trigger = "Yes" if p["will_trigger"] else "No"
        md_lines.append(f"| {p['name']} | {p['version']} | {p['priority']} | {trigger} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append(f"*Generated by DATADOC v{VERSION}*")
    md_lines.append("")

    output_path = f"report_{os.path.basename(file_path).split('.')[0]}.md"
    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))

    console.print()
    console.print(Panel(
        f"[bold green][OK] Report generated![/bold green]\n\n"
        f"  Saved to: [bold cyan]{output_path}[/bold cyan]",
        title="[bold]Report Export[/bold]",
        border_style="cyan",
    ))
    console.print()


# ──────────────────────────────────────────────────────────────
# COMMAND: plugin
# ──────────────────────────────────────────────────────────────
@app.command(name="plugin")
def plugin_list():
    """
    Lists all registered plugins and their status.
    """
    print_banner()



    from datadoc.plugins.missing_values import MissingValuePlugin
    from datadoc.plugins.outliers import OutlierPlugin
    from datadoc.plugins.datetime_feat import DatetimePlugin
    from datadoc.plugins.encoders import CategoricalEncoderPlugin
    from datadoc.plugins.scaling import ScalingPlugin

    plugins = sorted([
        MissingValuePlugin(),
        OutlierPlugin(),
        DatetimePlugin(),
        CategoricalEncoderPlugin(),
        ScalingPlugin(),
    ], key=lambda p: p.priority)

    console.print()
    table = Table(
        title="[bold]Registered Plugins[/bold]",
        box=box.ROUNDED,
        title_style="bold cyan",
        header_style="bold bright_white",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Priority", justify="center", style="yellow", width=10)
    table.add_column("Plugin", style="bold white", min_width=25)
    table.add_column("Version", justify="center", style="cyan", width=10)
    table.add_column("Description", style="dim white")

    for p in plugins:
        table.add_row(str(p.priority), p.name, p.version, p.description)

    console.print(table)

    # Plugin explanations
    console.print()
    console.print(Panel(
        "\n".join([f"  [bold cyan]{p.name}[/bold cyan]: {p.explain()}" for p in plugins]),
        title="[bold]Plugin Explanations[/bold]",
        border_style="dim",
    ))
    console.print()


# ──────────────────────────────────────────────────────────────
# COMMAND: version
# ──────────────────────────────────────────────────────────────
@app.command()
def version():
    """
    Displays the current DATADOC version.
    """
    print_banner()


# ──────────────────────────────────────────────────────────────
# COMMAND: chat
# ──────────────────────────────────────────────────────────────
@app.command()
def chat(
    file_path: str,
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="The LiteLLM model string to use.")
):
    """
    Start an interactive AI chat session with tool-calling capabilities.
    """
    import litellm
    import json
    from rich.markdown import Markdown
    from rich.prompt import Prompt

    print_banner()
    doc = load_dataset(file_path)
    api_key = _get_api_key(model)

    console.print()
    console.print(Panel(
        f"You are now chatting with the AI Assistant ({model}).\n"
        f"The AI has access to your dataset and can run plugins autonomously.\n"
        f"Type [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] to leave.",
        title="[bold green]Interactive Chat Session[/bold green]",
        border_style="green"
    ))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_dataset",
                "description": "Returns the current health report and metadata of the dataset.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "apply_plugin",
                "description": "Applies a specific dataset engineering plugin to the current dataset. You must use this to transform data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin_name": {
                            "type": "string",
                            "description": "Name of the plugin.",
                            "enum": ["MissingValuePlugin", "OutlierPlugin", "DatetimePlugin", "CategoricalEncoderPlugin", "ScalingPlugin"]
                        }
                    },
                    "required": ["plugin_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "revert_dataset",
                "description": "Reverts the dataset to its original loaded state, undoing all plugin transformations.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_dataset",
                "description": "Saves the current state of the dataset to a CSV file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the file to save (e.g., 'cleaned_data.csv')"
                        }
                    },
                    "required": ["filename"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": f"""You are a Principal Data Scientist assisting a user in an interactive chat session.
You have the ability to run dataset engineering tools.
Current dataset metadata: {doc._extract_metadata()}

When the user asks you to perform an action, use the appropriate tool.
Always explain what you did after using a tool. 
Be helpful, analytical, and concise. Format your responses with markdown."""
        }
    ]

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            if user_input.strip().lower() in ['exit', 'quit']:
                console.print("\n[bold green]Ending chat session. Goodbye![/bold green]\n")
                break
                
            messages.append({"role": "user", "content": user_input})
            
            with console.status("[bold cyan]AI is thinking...", spinner="dots"):
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    api_key=api_key
                )
                
                response_msg = response.choices[0].message
                messages.append(response_msg.model_dump(exclude_none=True))

                # Handle tool calls if any
                while response_msg.tool_calls:
                    for tool_call in response_msg.tool_calls:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        
                        console.print(f"  [dim]>[/dim] AI called tool: [bold yellow]{func_name}[/bold yellow]({args})")
                        
                        tool_result = ""
                        try:
                            if func_name == "analyze_dataset":
                                tool_result = json.dumps(doc.analyze(), indent=2)
                            elif func_name == "apply_plugin":
                                tool_result = doc.apply_plugin_by_name(args["plugin_name"])
                            elif func_name == "revert_dataset":
                                doc.revert()
                                tool_result = "Dataset reverted to original state."
                            elif func_name == "save_dataset":
                                doc.df.write_csv(args["filename"])
                                tool_result = f"Dataset successfully saved to {args['filename']}."
                            else:
                                tool_result = f"Error: Unknown function {func_name}"
                        except (ValueError, KeyError, RuntimeError) as e:
                            tool_result = f"Error executing {func_name}: {e}"
                            
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": tool_result
                        })
                        
                    # Call LLM again with tool results
                    response = litellm.completion(
                        model=model,
                        messages=messages,
                        tools=tools,
                        api_key=api_key
                    )
                    response_msg = response.choices[0].message
                    messages.append(response_msg.model_dump(exclude_none=True))
                
            # Print final assistant message
            if response_msg.content:
                console.print("\n[bold purple]DATADOC AI[/bold purple]:")
                console.print(Panel(Markdown(response_msg.content), border_style="purple"))

        except KeyboardInterrupt:
            console.print("\n[bold green]Ending chat session. Goodbye![/bold green]\n")
            break
        except (KeyboardInterrupt, RuntimeError) as e:
            console.print(f"\n[bold red]Error:[/] {e}")

if __name__ == "__main__":
    app()
