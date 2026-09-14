import json
import warnings
from typing import Any, Optional
import polars as pl
from rich.console import Console

console = Console()

from datadoc.core.pipeline import DataDocPipeline, PipelineConfig
from datadoc.ai.client import get_llm_client, is_ai_configured
from datadoc.ai.prompts import build_profile_digest, build_agent_prompt


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
                config.scaling = action.get("scaling", "auto")
            elif a_type == "AddInteraction":
                config.custom_interactions.append(action)
            elif a_type == "ApplyTransform":
                config.custom_transforms.append(action)
            elif a_type == "IgnoreColumn":
                config.ignored_columns.append(action.get("col", ""))

    def _evaluate_and_prune(self, df: pl.DataFrame, config: PipelineConfig) -> dict[str, float]:
        """Runs a quick Random Forest fit to get feature importances."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError as e:
            raise RuntimeError("pip install 'datadoc-cli[ml]' for agent evaluation.") from e

        # Fit the pipeline
        pipeline = DataDocPipeline(config).fit(df, target=self.target)
        transformed = pipeline.transform(df)

        feature_columns = [
            name for name, dtype in transformed.schema.items() 
            if name != self.target and dtype.is_numeric()
        ]

        if not feature_columns:
            return {}

        train_features = transformed.select(feature_columns)
        # Impute missing values with 0.0 for quick RF fitting
        train_features = train_features.fill_null(0.0).fill_nan(0.0)
        
        x_train = train_features.to_numpy()
        y_train = df[self.target].to_numpy()

        target_dtype = df[self.target].dtype
        is_classification = not (target_dtype.is_numeric() and df[self.target].n_unique() > 20)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=config.random_seed)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=config.random_seed)
            model.fit(x_train, y_train)

        importances = dict(zip(feature_columns, model.feature_importances_))
        return importances

    def run_interactive(self, df: pl.DataFrame, iterations: int = 1, interactive: bool = False) -> PipelineConfig:
        console.print(f"\n[bold cyan]🚀 Starting DATADOC Agentic Loop for target: '{self.target}'[/bold cyan]")
        
        # Initial Profile
        pipeline = DataDocPipeline(PipelineConfig(target=self.target))
        profile = pipeline.profile(df)
        digest = build_profile_digest(profile.to_dict(), df, self.target)

        current_config = PipelineConfig(target=self.target)

        for i in range(iterations):
            console.print(f"\n[bold magenta]--- Iteration {i+1}/{iterations} ---[/bold magenta]")
            
            # 1. Interactive Interview
            if interactive:
                console.print("\n[bold]Agent:[/bold] I have analyzed the profile. What domain knowledge can you share about these columns?")
                user_input = input("You: ")
            else:
                user_input = ""
            if not user_input.strip():
                user_input = "No domain knowledge provided. Proceed with best statistical practices."
            
            self.chat_history.append(f"User: {user_input}")

            # 2. Generate Hypotheses
            console.print("\n[dim]Generating hypotheses...[/dim]")
            chat_context = "\n".join(self.chat_history)
            prompt = build_agent_prompt(digest, self.target, chat_context)
            
            # We inject the JSON schema into the prompt to guide the LLM
            schema_instructions = """
You must return a JSON array containing objects with a "type" field. 
Valid types: ToggleFeature, SetScaling, AddInteraction, ApplyTransform, IgnoreColumn.
Example:
[
  {"type": "ApplyTransform", "col": "income", "transform": "log1p"},
  {"type": "AddInteraction", "col_a": "price", "col_b": "sqft", "op": "div"}
]
"""
            prompt += schema_instructions
            
            response = self.client.complete(prompt)
            actions = self._extract_json_array(response)

            if not actions:
                console.print("[yellow]No valid actions generated. Skipping iteration.[/yellow]")
                continue
            
            console.print(f"[cyan]Agent proposed {len(actions)} feature engineering actions.[/cyan]")
            
            # 3. Apply Actions to a temporary config
            import copy
            candidate_config = copy.deepcopy(current_config)
            self._apply_actions(candidate_config, actions)

            # 4. Evaluate and Prune
            console.print("[dim]Evaluating via Mathematical Referee (Random Forest)...[/dim]")
            baseline_importances = self._evaluate_and_prune(df, current_config)
            candidate_importances = self._evaluate_and_prune(df, candidate_config)

            # Very simple pruning: if a newly created column has 0.0 importance, we drop the action.
            # In a robust implementation, we would evaluate SHAP or drop actions that lower the CV score.
            # Here, we do a basic CV test using pipeline's built-in evaluator to check if score improves.
            
            baseline_report = DataDocPipeline(current_config).evaluate(df, target=self.target)
            candidate_report = DataDocPipeline(candidate_config).evaluate(df, target=self.target)

            improvement = candidate_report.improvement
            if improvement > 0:
                console.print(f"[green]✅ Batch accepted! {candidate_report.metric} improved by {improvement:.4f}[/green]")
                current_config = candidate_config
                self.chat_history.append(f"Agent: Batch accepted. Improvement: {improvement:.4f}")
            else:
                console.print(f"[red]❌ Batch rejected. {candidate_report.metric} worsened by {improvement:.4f}. Pruning.[/red]")
                self.chat_history.append(f"Agent: Batch rejected. Improvement: {improvement:.4f}")

        console.print("\n[bold green]🎉 Agentic Loop Complete. Final Plan generated.[/bold green]")
        return current_config
