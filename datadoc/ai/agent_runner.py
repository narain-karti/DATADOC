import copy
import json
import warnings
from typing import Any, Optional
import polars as pl
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(legacy_windows=False)

from datadoc.core.pipeline import DataDocPipeline, PipelineConfig
from datadoc.ai.client import get_llm_client, is_ai_configured
from datadoc.ai.prompts import build_profile_digest, build_agent_prompt


def _format_action_details(action: dict[str, Any]) -> str:
    """Format an action into a clean human-readable expression."""
    a_type = action.get("type", "")
    if a_type == "AddInteraction":
        col_a = action.get("col_a", "?")
        col_b = action.get("col_b", "?")
        op = action.get("op", "?")
        op_sym = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}.get(op, op)
        return f"{col_a} {op_sym} {col_b}"
    elif a_type == "ApplyTransform":
        col = action.get("col", "?")
        tf = action.get("transform", "?")
        return f"{tf}({col})"
    elif a_type == "SetScaling":
        return f"scaling: {action.get('scaling', 'auto')}"
    elif a_type == "ToggleFeature":
        return f"{action.get('setting', '?')} = {action.get('value', '?')}"
    elif a_type == "IgnoreColumn":
        return f"drop {action.get('col', '?')}"
    return str(action)


class DataDocAgent:
    """The interactive Hypothesis Generator and Mathematical Referee."""

    def __init__(self, target: str, model: Optional[str] = None):
        if not is_ai_configured():
            raise RuntimeError(
                "No AI API key found. Please set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )
        self.target = target
        self.client = get_llm_client(model)
        self.chat_history: list[str] = []

    def _extract_json_array(self, text: str) -> list[dict[str, Any]]:
        """Safely extract a JSON array from the LLM's raw text response."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "actions" in parsed:
                return parsed["actions"]
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            console.print("[red]Failed to parse JSON from LLM. Raw output:[/red]")
            console.print(text)
            return []

    def _apply_actions(self, config: PipelineConfig, actions: list[dict[str, Any]]) -> None:
        """Apply the JSON actions to the PipelineConfig."""
        for action in actions:
            a_type = action.get("type", "")
            if a_type == "ToggleFeature":
                setting = action.get("setting")
                value = action.get("value")
                if hasattr(config, setting):
                    setattr(config, setting, bool(value))
            elif a_type == "SetScaling":
                val = action.get("scaling", "auto")
                if val in ("none", "standard", "robust", "auto"):
                    config.scaling = val
            elif a_type == "AddInteraction":
                if action not in config.custom_interactions:
                    config.custom_interactions.append(action)
            elif a_type == "ApplyTransform":
                if action not in config.custom_transforms:
                    config.custom_transforms.append(action)
            elif a_type == "IgnoreColumn":
                col = action.get("col", "")
                if col and col not in config.ignored_columns:
                    config.ignored_columns.append(col)

    def run_interactive(
        self, df: pl.DataFrame, iterations: int = 1, interactive: bool = False
    ) -> PipelineConfig:
        # 1. Detect dataset task and statistics
        target_dtype = df[self.target].dtype
        is_classification = not (target_dtype.is_numeric() and df[self.target].n_unique() > 20)
        if is_classification:
            n_classes = df[self.target].n_unique()
            task_type = f"Binary Classification ({n_classes} classes)" if n_classes == 2 else f"Multiclass Classification ({n_classes} classes)"
            referee_desc = "RandomForestClassifier (50 trees, 3-Fold Stratified CV)"
        else:
            task_type = "Continuous Regression"
            referee_desc = "RandomForestRegressor (50 trees, 3-Fold CV)"

        # 2. Display Session Overview Banner
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="bold cyan")
        info_table.add_column(style="white")
        info_table.add_row("Dataset:", f"{df.height:,} rows × {df.width} columns")
        info_table.add_row("Target Column:", f"[bold yellow]{self.target}[/bold yellow] ({task_type})")
        info_table.add_row("Referee Model:", f"[green]{referee_desc}[/green]")
        info_table.add_row("Engine:", "[blue]Declarative Leakage-Safe Polars/Rust Pipeline[/blue]")
        info_table.add_row("Mode:", "[magenta]Interactive Co-Pilot[/magenta]" if interactive else "[cyan]Autonomous Auto-Research[/cyan]")

        console.print(
            Panel(
                info_table,
                title="[bold cyan]🤖 DATADOC Autonomous Feature Discovery Agent[/bold cyan]",
                subtitle="[dim]Continuous Hypothesis Generation & Mathematical Referee Loop[/dim]",
                border_style="cyan",
            )
        )

        # 3. Initial Profile Digest
        pipeline = DataDocPipeline(PipelineConfig(target=self.target))
        profile = pipeline.profile(df)
        digest = build_profile_digest(profile.to_dict(), df, self.target)

        # 4. Interactive Confirmation / Domain Intuition Prompt
        if interactive:
            console.print("\n[bold cyan]Agent:[/bold cyan] I have analyzed the profile. What domain knowledge or column relationships should I investigate?")
            console.print("[dim]Tip: Press Enter to let the agent auto-explore statistical hypotheses without hints.[/dim]")
            user_input = Prompt.ask("[bold green]You[/bold green]", default="")
            if user_input.strip():
                self.chat_history.append(f"Domain Expert guidance: {user_input.strip()}")
                console.print(f"[dim]✓ Domain guidance recorded: '{user_input.strip()}'[/dim]\n")
            else:
                self.chat_history.append("Domain Expert: No initial hints. Explore statistical interactions autonomously.")
                console.print("[dim]✓ Proceeding with autonomous exploration.[/dim]\n")

        # 5. Baseline Evaluation Benchmark
        current_config = PipelineConfig(target=self.target, estimator_family="tree")
        with console.status("[bold yellow]Mathematical Referee: Computing initial baseline benchmark...[/bold yellow]", spinner="dots"):
            baseline_report = DataDocPipeline(current_config).evaluate(df, target=self.target)

        current_score = baseline_report.selected_score
        initial_score = current_score
        metric_name = baseline_report.metric.replace("_", " ").title()

        console.print(f"[bold]Initial Baseline Benchmark ({metric_name}):[/bold] [bold yellow]{current_score:.4f}[/bold yellow]\n")

        # 6. Iterative Auto-Research Loop
        for i in range(1, iterations + 1):
            console.print(Rule(f"[bold magenta]Iteration {i}/{iterations}[/bold magenta]", style="magenta"))

            # In interactive mode, ask contextual follow-ups after round 1
            if interactive and i > 1:
                active_count = len(current_config.custom_interactions) + len(current_config.custom_transforms)
                console.print(
                    f"\n[bold cyan]Agent:[/bold cyan] Current best [bold]{metric_name}[/bold] is [green]{current_score:.4f}[/green] ({active_count} custom features active)."
                )
                console.print("Do you have any adjustments or new relationships to suggest for this round? [dim](Press Enter to continue auto-discovery)[/dim]")
                user_followup = Prompt.ask("[bold green]You[/bold green]", default="")
                if user_followup.strip():
                    self.chat_history.append(f"Domain Expert (Round {i}): {user_followup.strip()}")
                    console.print(f"[dim]✓ Feedback recorded: '{user_followup.strip()}'[/dim]\n")
                else:
                    self.chat_history.append(f"Domain Expert (Round {i}): Continue autonomous discovery.")
                    console.print("[dim]✓ Continuing autonomous discovery.[/dim]\n")

            # A. Formulate Hypotheses with Live Spinner
            with console.status(
                f"[bold cyan]Agent is formulating ML hypotheses & feature transformations (Round {i}/{iterations})...[/bold cyan]",
                spinner="dots",
            ):
                chat_context = "\n".join(self.chat_history)
                prompt = build_agent_prompt(digest, self.target, chat_context)
                response = self.client.complete(prompt)
                actions = self._extract_json_array(response)

            if not actions:
                console.print("[yellow]Agent proposed no valid actions this round. Skipping.[/yellow]\n")
                continue

            # B. Display Proposed Actions Table
            actions_table = Table(
                title=f"Proposed Feature Engineering Actions (Round {i})",
                border_style="cyan",
                header_style="bold cyan",
                show_lines=True,
            )
            actions_table.add_column("#", style="dim", width=4, justify="center")
            actions_table.add_column("Type", style="bold yellow", width=16)
            actions_table.add_column("Operation / Expression", style="bold white", width=28)
            actions_table.add_column("Domain Rationale", style="white")

            for idx, act in enumerate(actions, 1):
                raw_rationale = act.get("rationale") or "Domain hypothesis for predictive lift"
                actions_table.add_row(
                    str(idx),
                    escape(act.get("type", "Action")),
                    escape(_format_action_details(act)),
                    escape(raw_rationale),
                )
            console.print(actions_table)

            # C. Apply Actions to Candidate Config
            candidate_config = copy.deepcopy(current_config)
            self._apply_actions(candidate_config, actions)

            # D. Evaluate Candidate via Mathematical Referee with Live Spinner
            with console.status(
                "[bold yellow]Mathematical Referee: Running 3-Fold Cross-Validation & Out-of-Fold Evaluation...[/bold yellow]",
                spinner="bouncingBar",
            ):
                candidate_report = DataDocPipeline(candidate_config).evaluate(df, target=self.target)

            cand_score = candidate_report.selected_score
            delta = cand_score - current_score
            pct_change = (delta / abs(current_score) * 100) if current_score != 0 else 0.0
            total_active_feats = len(candidate_config.custom_interactions) + len(candidate_config.custom_transforms)

            # E. Render Scorecard Table
            score_table = Table(
                title=f"Referee Benchmark Scorecard (Round {i})",
                border_style="green" if delta > 0 else "red",
                header_style="bold",
                show_lines=True,
            )
            score_table.add_column("Metric", style="bold white")
            score_table.add_column("Previous Best", justify="right")
            score_table.add_column("Candidate Batch", justify="right")
            score_table.add_column("Net Lift (Δ)", justify="right")
            score_table.add_column("Decision", justify="center")
            score_table.add_column("Active Custom Features", justify="center")

            if delta > 0:
                decision_str = "[bold green]✅ ACCEPTED (Merged)[/bold green]"
                lift_str = f"[bold green]+{delta:.4f} (+{pct_change:.2f}%)[/bold green]"
                current_config = candidate_config
                current_score = cand_score
                self.chat_history.append(
                    f"Round {i}: Accepted {len(actions)} actions. {metric_name} improved from {current_score - delta:.4f} to {current_score:.4f} (+{delta:.4f})."
                )
                score_table.add_row(
                    metric_name,
                    f"{current_score - delta:.4f}",
                    f"{cand_score:.4f}",
                    lift_str,
                    decision_str,
                    str(total_active_feats),
                )
                console.print(score_table)
                console.print(f"[green]✓ Batch accepted! New baseline {metric_name}: [bold]{current_score:.4f}[/bold][/green]\n")
            else:
                decision_str = "[bold red]❌ REJECTED (Pruned)[/bold red]"
                lift_str = f"[bold red]{delta:.4f} ({pct_change:.2f}%)[/bold red]"
                self.chat_history.append(
                    f"Round {i}: Rejected candidate batch. {metric_name} changed by {delta:.4f}. Pruned to prevent regression."
                )
                score_table.add_row(
                    metric_name,
                    f"{current_score:.4f}",
                    f"{cand_score:.4f}",
                    lift_str,
                    decision_str,
                    str(len(current_config.custom_interactions) + len(current_config.custom_transforms)),
                )
                console.print(score_table)
                console.print(f"[red]✗ Batch rejected to protect against overfitting/leakage. Retained baseline: [bold]{current_score:.4f}[/bold][/red]\n")

        # 7. Final Summary
        total_delta = current_score - initial_score
        total_pct = (total_delta / abs(initial_score) * 100) if initial_score != 0 else 0.0

        lift_color = "bold green" if total_delta >= 0 else "bold red"
        summary_lines = [
            f"[bold]Initial Baseline ({metric_name}):[/bold] {initial_score:.4f}",
            f"[bold]Final Pipeline ({metric_name}):[/bold] [bold green]{current_score:.4f}[/bold green]",
            f"[bold]Total Net Lift:[/bold] [{lift_color}]{total_delta:+.4f} ({total_pct:+.2f}%)[/{lift_color}]",
            f"[bold]Discovered Interactions:[/bold] {len(current_config.custom_interactions)}",
            f"[bold]Discovered Transforms:[/bold] {len(current_config.custom_transforms)}",
            f"[bold]Scaling Strategy:[/bold] {current_config.scaling}",
            f"[bold]Outlier Handling:[/bold] {'IQR Clipping enabled' if current_config.clip_outliers else 'None'}",
            f"[bold]Missing Value Indicators:[/bold] {'Enabled' if current_config.add_missing_indicators else 'Disabled'}",
        ]

        console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold green]🎉 Agentic Auto-Research Complete[/bold green]",
                subtitle="[dim]Final leakage-safe pipeline configuration synthesized[/dim]",
                border_style="green",
            )
        )
        return current_config
