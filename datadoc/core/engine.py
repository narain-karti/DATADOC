import polars as pl
from typing import Dict, Any, List
import json
from collections import Counter
import litellm
from pydantic import BaseModel, Field
from datadoc.core.agent import AgenticEngineer
from datadoc.plugins.base import BasePlugin
from datadoc.plugins.missing_values import MissingValuePlugin
from datadoc.plugins.outliers import OutlierPlugin
from datadoc.plugins.datetime_feat import DatetimePlugin
from datadoc.plugins.encoders import CategoricalEncoderPlugin
from datadoc.plugins.scaling import ScalingPlugin

class DATADOC:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pl.read_csv(file_path, infer_schema_length=10000)
        self._original_df = self.df.clone()

        # Plugins sorted by priority (lower = runs first)
        self.plugins: List[BasePlugin] = sorted([
            MissingValuePlugin(),
            OutlierPlugin(),
            DatetimePlugin(),
            CategoricalEncoderPlugin(),
            ScalingPlugin(),
        ], key=lambda p: p.priority)

    @staticmethod
    def _detect_column_roles(df: pl.DataFrame) -> Dict[str, str]:
        """Classify each column as 'id', 'name', 'constant', or 'feature'."""
        roles = {}
        for col in df.columns:
            series = df[col]
            # Constant columns (single unique value)
            if series.drop_nulls().n_unique() <= 1:
                roles[col] = "constant"
            # Likely ID: integer, all unique, monotonically increasing
            elif series.dtype.is_integer() and series.drop_nulls().n_unique() == series.drop_nulls().len() and series.drop_nulls().len() > 0:
                sorted_vals = series.drop_nulls().sort()
                diffs = sorted_vals.diff().drop_nulls()
                if diffs.len() > 0 and (diffs == diffs[0]).all():
                    roles[col] = "id"
                else:
                    roles[col] = "feature"
            # String columns need extra checks
            elif series.dtype == pl.String and series.drop_nulls().n_unique() == series.drop_nulls().len() and series.drop_nulls().len() > 0:
                # Check if it's a datetime string before labelling as name
                try:
                    parsed = df.select(pl.col(col).str.to_datetime(strict=False))
                    not_null_ratio = 1 - (parsed[col].null_count() / max(parsed.height, 1))
                    if not_null_ratio > 0.5:
                        roles[col] = "feature"  # Datetime string — let DatetimePlugin handle it
                    else:
                        roles[col] = "name"
                except (ValueError, TypeError, pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
                    roles[col] = "name"
            else:
                roles[col] = "feature"
        return roles

    @staticmethod
    def _drop_constant_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        """Drop columns with zero variance (single unique non-null value)."""
        to_drop = []
        for col in df.columns:
            if df[col].drop_nulls().n_unique() <= 1:
                to_drop.append(col)
        if to_drop:
            df = df.drop(to_drop)
        return df, to_drop

    def analyze(self) -> Dict[str, Any]:
        """Runs the analyze step across all plugins."""
        dtype_counts = Counter([str(dt) for dt in self.df.dtypes])
        report = {
            "rows": self.df.height,
            "cols": self.df.width,
            "dtypes": dict(dtype_counts),
            "plugins": {}
        }

        for plugin in self.plugins:
            report["plugins"][plugin.name] = plugin.analyze(self.df)

        return report

    def recommend(self) -> list[str]:
        """Runs analysis and aggregates recommendations from all plugins."""
        recommendations = []
        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            recs = plugin.recommend(analysis)
            recommendations.extend(recs)
        return recommendations

    def pipeline(self) -> str:
        """Generates the full Python script representing the pipeline."""
        lines = [
            "import polars as pl",
            "import numpy as np",
            "",
            "",
            f"def load_and_clean_data(file_path: str = '{self.file_path}') -> pl.DataFrame:",
            "    df = pl.read_csv(file_path, infer_schema_length=10000)",
            "",
        ]

        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            should_apply = any(bool(v) for k, v in analysis.items() if k.startswith('has_'))
            if should_apply:
                code_snippet = plugin.generate_code(analysis)
                if code_snippet:
                    # Indent each line of the code snippet
                    for line in code_snippet.split("\n"):
                        lines.append(f"    {line}")
                    lines.append("")

        lines.append("    return df")
        lines.append("")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    clean_df = load_and_clean_data()")
        lines.append("    print(f'Successfully processed {clean_df.height} rows x {clean_df.width} columns!')")
        lines.append("")

        return "\n".join(lines)

    def engineer(self, progress_callback=None) -> pl.DataFrame:
        """Automatically triggers plugins that are needed."""
        df_transformed = self.df.clone()

        self._applied_plugins = []
        self._skipped_plugins = []
        self._dropped_columns = []

        # Phase 0: Drop ID and name columns before any plugin runs
        roles = self._detect_column_roles(df_transformed)
        cols_to_drop = [c for c, r in roles.items() if r in ("id", "name")]
        if cols_to_drop:
            df_transformed = df_transformed.drop(cols_to_drop)
            self._dropped_columns.extend(cols_to_drop)
            if progress_callback:
                progress_callback("ColumnFilter", "applied",
                    [f"Dropped non-feature columns: {', '.join(cols_to_drop)}"])

        for plugin in self.plugins:
            if progress_callback:
                progress_callback(plugin.name, "running", [])
                
            analysis = plugin.analyze(df_transformed)
            should_apply = any(bool(v) for k, v in analysis.items() if k.startswith('has_'))
            if should_apply:
                details = plugin.recommend(analysis)
                df_transformed = plugin.apply(df_transformed)
                self._applied_plugins.append(plugin.name)
                if progress_callback:
                    progress_callback(plugin.name, "applied", details)
            else:
                self._skipped_plugins.append(plugin.name)
                if progress_callback:
                    progress_callback(plugin.name, "skipped", [])

        # Phase final: Drop any constant columns created during pipeline
        df_transformed, dropped_constants = self._drop_constant_columns(df_transformed)
        if dropped_constants:
            self._dropped_columns.extend(dropped_constants)
            if progress_callback:
                progress_callback("ColumnFilter", "applied",
                    [f"Dropped constant columns: {', '.join(dropped_constants)}"])

        return df_transformed

    def compare(self, clean_df: pl.DataFrame) -> Dict[str, Any]:
        """Compare original and engineered dataframes."""
        original = self.df
        
        orig_missing = sum(original[c].null_count() for c in original.columns)
        clean_missing = sum(clean_df[c].null_count() for c in clean_df.columns)
        
        result = {
            "original_shape": original.shape,
            "clean_shape": clean_df.shape,
            "rows_changed": original.height != clean_df.height,
            "cols_added": clean_df.width - original.width,
            "original_missing": orig_missing,
            "clean_missing": clean_missing,
            "original_dtypes": dict(Counter([str(dt) for dt in original.dtypes])),
            "clean_dtypes": dict(Counter([str(dt) for dt in clean_df.dtypes])),
        }
        return result

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Returns info about all registered plugins."""
        info = []
        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            has_work = any(bool(v) for k, v in analysis.items() if k.startswith('has_'))
            info.append({
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "priority": plugin.priority,
                "will_trigger": has_work,
                "dependencies": plugin.dependencies,
            })
        return info

    def revert(self) -> None:
        """Reverts the dataset to its original loaded state."""
        self.df = self._original_df.clone()

    def apply_plugin_by_name(self, plugin_name: str) -> str:
        """Applies a specific plugin by name and updates the internal dataframe."""
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                self.df = plugin.apply(self.df)
                return f"Successfully applied {plugin_name}."
        return f"Error: Plugin {plugin_name} not found."

    def _extract_metadata(self) -> str:
        """Safely extracts dataset metadata without exposing raw rows."""
        meta = {
            "rows": self.df.height,
            "columns": self.df.width,
            "schema": {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)},
            "null_counts": {col: self.df[col].null_count() for col in self.df.columns if self.df[col].null_count() > 0},
            "available_plugins": [p.name for p in self.plugins]
        }
        return json.dumps(meta, indent=2)

    def _generate_ai_plan(self, model: str, goal: str, api_key: str = None):
        """Helper to generate an AI plan."""
        class PluginExecution(BaseModel):
            plugin_name: str = Field(description="The exact class name of the plugin to execute.")
            reason: str = Field(description="A highly detailed, in-depth analytical explanation of exactly why this transformation is necessary given the specific dataset metadata and how it directly supports the user's goal. Explain the 'why' thoroughly.")
            
        class AIPlannerResponse(BaseModel):
            plan: List[PluginExecution]

        metadata = self._extract_metadata()
        
        prompt = f"""You are a Principal Data Scientist building a robust dataset engineering pipeline.
You have the following detailed dataset metadata:
{metadata}

The user's specific goal is: "{goal}"

Based on the metadata and the goal, create an execution plan selecting ONLY from the `available_plugins`.
You must provide deep analytical reasoning for every plugin you select. Do not be vague. Explain exactly what data health issues are present and how the selected plugin solves them in the context of the user's goal (e.g. XGBoost handles missing values differently than standard Regression, scaling is vital for distance-based models but not trees, etc.).
You may skip plugins if they are not relevant to the goal. You may order them logically.
Respond strictly with a JSON object matching the requested schema.
"""

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            response_format=AIPlannerResponse
        )
        
        plan_data = json.loads(response.choices[0].message.content)
        return AIPlannerResponse(**plan_data)

    def ai_analyze(self, model: str, api_key: str = None) -> str:
        """Uses an LLM to generate an in-depth Executive Summary of the dataset's health."""
        metadata = self._extract_metadata()
        prompt = f"""You are a Principal Data Scientist performing a rigorous dataset health audit. Review the following dataset metadata:
{metadata}

Provide an in-depth, detailed diagnostic report of the dataset's health. 
Analyze the schema, identify any architectural problems, explicitly call out critical issues (like missing values, outliers, high cardinality, or scaling imbalances), and explain the potential downstream impact on Machine Learning models if left untreated.
Do not be vague. Be extremely specific based on the provided metadata. 
Respond in plain text formatted nicely with bullet points and paragraphs where appropriate.
"""
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key
        )
        return response.choices[0].message.content.strip()

    def ai_recommend(self, model: str, goal: str, api_key: str = None) -> List[Dict[str, str]]:
        """Uses an LLM to generate an execution plan for recommendations."""
        planner_response = self._generate_ai_plan(model, goal, api_key)
        return [{"plugin": step.plugin_name, "reason": step.reason} for step in planner_response.plan]

    def ai_engineer(self, model: str, goal: str, api_key: str = None, progress_callback=None) -> pl.DataFrame:
        """Uses an LLM to plan and execute a custom pipeline based on a user goal."""
        if progress_callback:
            progress_callback("AI Planner", "running", ["Consulting LLM..."])
            
        try:
            planner_response = self._generate_ai_plan(model, goal, api_key)
        except (RuntimeError, ValueError, TypeError) as e:
            if progress_callback:
                progress_callback("AI Planner", "error", [f"LLM API Error: {e}"])
            raise RuntimeError(f"AI Planner failed: {e}")

        df_transformed = self.df.clone()
        self._applied_plugins = []
        self._skipped_plugins = []
        self._dropped_columns = []

        # Phase 0: Drop ID and name columns before any plugin runs
        roles = self._detect_column_roles(df_transformed)
        cols_to_drop = [c for c, r in roles.items() if r in ("id", "name")]
        if cols_to_drop:
            df_transformed = df_transformed.drop(cols_to_drop)
            self._dropped_columns.extend(cols_to_drop)
            if progress_callback:
                progress_callback("ColumnFilter", "applied",
                    [f"Dropped non-feature columns: {', '.join(cols_to_drop)}"])

        if progress_callback:
            progress_callback("AI Planner", "applied", [f"Generated plan with {len(planner_response.plan)} steps."])

        # Execute the plan
        plugin_map = {p.name: p for p in self.plugins}
        
        for step in planner_response.plan:
            if step.plugin_name not in plugin_map:
                continue
                
            plugin = plugin_map[step.plugin_name]
            
            if progress_callback:
                progress_callback(plugin.name, "running", [step.reason])
                
            df_transformed = plugin.apply(df_transformed)
            self._applied_plugins.append(plugin.name)
            
            if progress_callback:
                progress_callback(plugin.name, "applied", [f"Reason: {step.reason}"])
                
        # Find skipped plugins
        for p in self.plugins:
            if p.name not in self._applied_plugins:
                self._skipped_plugins.append(p.name)
                if progress_callback:
                    progress_callback(p.name, "skipped", ["Skipped by AI Planner"])

        # Phase final: Drop any constant columns created during pipeline
        df_transformed, dropped_constants = self._drop_constant_columns(df_transformed)
        if dropped_constants:
            self._dropped_columns.extend(dropped_constants)
            if progress_callback:
                progress_callback("ColumnFilter", "applied",
                    [f"Dropped constant columns: {', '.join(dropped_constants)}"])
                    
        return df_transformed

    def agentic_engineer(self, model: str, goal: str, api_key: str = None, interactive: bool = True) -> pl.DataFrame:
        """
        Launches the Autonomous AI Data Engineer to plan, write, and execute custom pipeline code.
        If interactive=True, it will ask questions via input().
        If interactive=False, it will execute autonomously based on the goal.
        """
        metadata = self._extract_metadata()
        agent = AgenticEngineer(metadata=metadata, api_key=api_key, model=model)
        
        print("\n🤖 DATADOC Agentic Engineer initializing...")
        
        # 1. Interview Phase
        if interactive:
            print("\n[AI]: " + agent.chat_step(""))
            while True:
                user_msg = input("\n[You]: ")
                if user_msg.strip().lower() in ['plan', 'go', 'execute', 'done', 'yes', 'y']:
                    break
                print("\n[AI]: " + agent.chat_step(user_msg))
        else:
            agent.chat_step(f"My goal is: {goal}. Please generate the plan.")
            
        # 2. Plan Phase
        print("\n⚙️ Generating Implementation Plan...")
        plan = agent.generate_plan()
        print("\n=== AI IMPLEMENTATION PLAN ===\n" + plan + "\n==============================\n")
        
        if interactive:
            approve = input("Do you approve this plan? (Y/N): ")
            if approve.strip().lower() not in ['y', 'yes']:
                print("Aborting.")
                return self.df
                
        # 3. Code Generation Phase
        print("\n💻 Generating Custom Pipeline Code...")
        code = agent.generate_code()
        self.last_agent_code = code
        print("\n=== GENERATED PIPELINE ===\n" + code + "\n==========================\n")
        
        if interactive:
            approve_code = input("Do you want to execute this code now? (Y/N): ")
            if approve_code.strip().lower() not in ['y', 'yes']:
                print("Execution aborted. The code is saved in `doc.last_agent_code`.")
                return self.df
                
        # 4. Execution Phase
        print("\n🚀 Executing Custom Pipeline...")
        # Create a safe local namespace to execute the function
        local_vars = {}
        try:
            exec(code, globals(), local_vars)
            if 'clean_data' not in local_vars:
                raise ValueError("The AI failed to generate a `clean_data` function.")
                
            clean_func = local_vars['clean_data']
            # Pass the dataframe to the function
            clean_df = clean_func(self.df.clone())
            
            # Ensure it returned a dataframe
            if not isinstance(clean_df, (pl.DataFrame, type(pd.DataFrame()))):
                raise TypeError(f"Expected a DataFrame, but got {type(clean_df)}")
                
            # Convert back to polars if it was pandas
            if not isinstance(clean_df, pl.DataFrame):
                clean_df = pl.from_pandas(clean_df)
                
            print("✅ Execution Successful!")
            return clean_df
            
        except Exception as e:
            print(f"\n❌ Execution Failed: {e}")
            print("The generated code might contain errors. You can inspect it via `doc.last_agent_code`.")
            return self.df
            
    def export_agent_pipeline(self, filename: str) -> None:
        """Saves the last generated AI pipeline code to a file."""
        if not hasattr(self, 'last_agent_code') or not self.last_agent_code:
            raise ValueError("No AI code has been generated yet. Run `agentic_engineer()` first.")
            
        with open(filename, 'w') as f:
            f.write(self.last_agent_code)
        print(f"✅ Pipeline exported to {filename}")

