import polars as pl
from typing import Dict, Any, List, Optional
import json
from collections import Counter
import litellm
from pydantic import BaseModel, Field
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

    def ai_engineer(self, model: str, goal: str, api_key: str = None, progress_callback=None) -> pl.DataFrame:
        """Uses an LLM to plan and execute a custom pipeline based on a user goal."""
        import os
        if api_key:
            # For litellm, we can just set the env var for the provider, 
            # but litellm handles api_key directly in completion if specified.
            # However, litellm often expects the specific provider key (e.g., GEMINI_API_KEY).
            # To be safe, we pass api_key to completion.
            pass
        
        class PluginExecution(BaseModel):
            plugin_name: str = Field(description="The exact class name of the plugin to execute.")
            reason: str = Field(description="Why this plugin is needed for the user's goal.")
            
        class AIPlannerResponse(BaseModel):
            plan: List[PluginExecution]

        metadata = self._extract_metadata()
        
        prompt = f"""You are an expert Data Scientist building a dataset engineering pipeline.
You have the following dataset metadata:
{metadata}

The user's specific goal is: "{goal}"

Based on the metadata and the goal, create an execution plan selecting ONLY from the `available_plugins`.
You may skip plugins if they are not relevant to the goal. You may order them logically.
Respond strictly with a JSON object matching the requested schema.
"""

        if progress_callback:
            progress_callback("AI Planner", "running", ["Consulting LLM..."])
            
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                response_format=AIPlannerResponse
            )
        except Exception as e:
            if progress_callback:
                progress_callback("AI Planner", "error", [f"LLM API Error ({e.__class__.__name__}): {e}"])
            raise RuntimeError(f"AI Planner failed: {e}")
        
        # Parse the JSON response
        try:
            plan_data = json.loads(response.choices[0].message.content)
            planner_response = AIPlannerResponse(**plan_data)
        except Exception as e:
            if progress_callback:
                progress_callback("AI Planner", "error", [f"Failed to parse LLM response: {e}"])
            raise ValueError(f"Invalid AI response format: {e}")

        df_transformed = self.df.clone()
        self._applied_plugins = []
        self._skipped_plugins = []

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
                    
        return df_transformed

