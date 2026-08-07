"""Compatibility facade for the legacy DATADOC API.

New applications should use :class:`datadoc.core.pipeline.DataDocPipeline`.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any
import warnings

import polars as pl

from datadoc.core.pipeline import (
    DataDocError,
    DataDocPipeline,
    PipelineConfig,
    profile_dataset,
    read_dataset,
)
from datadoc.plugins.datetime_feat import DatetimePlugin
from datadoc.plugins.encoders import CategoricalEncoderPlugin
from datadoc.plugins.missing_values import MissingValuePlugin
from datadoc.plugins.outliers import OutlierPlugin
from datadoc.plugins.scaling import ScalingPlugin


class DATADOC:
    """Deprecated mutable facade retained for backwards compatibility.

    It creates a fitted :class:`DataDocPipeline` internally. New code should
    construct ``DataDocPipeline(PipelineConfig(...))`` directly.
    """

    def __init__(
        self,
        file_path: str,
        categorical_threshold: int = 10,
        scaling_ratio: float = 10.0,
        outlier_multiplier: float = 1.5,
    ):
        self.file_path = file_path
        self.df = read_dataset(file_path)
        self._original_df = self.df.clone()
        self.categorical_threshold = categorical_threshold
        self.scaling_ratio = scaling_ratio
        self.outlier_multiplier = outlier_multiplier
        self.plugins = [
            MissingValuePlugin(),
            OutlierPlugin(outlier_multiplier),
            DatetimePlugin(),
            CategoricalEncoderPlugin(categorical_threshold),
            ScalingPlugin(scaling_ratio),
        ]
        self._applied_plugins: list[str] = []
        self._skipped_plugins: list[str] = []
        self._dropped_columns: list[str] = []
        self.fitted_pipeline: DataDocPipeline | None = None

    @staticmethod
    def _detect_column_roles(df: pl.DataFrame) -> dict[str, str]:
        profile = profile_dataset(df, PipelineConfig())
        mapping = {
            "feature_numeric": "feature",
            "feature_categorical": "feature",
            "feature_datetime": "feature",
            "identifier": "id",
            "text": "name",
            "constant": "constant",
            "ignored": "feature",
            "target": "feature",
        }
        result = {role.name: mapping[role.role] for role in profile.roles}
        for role in profile.roles:
            if (
                role.role == "identifier"
                and df[role.name].dtype == pl.String
                and "name" in role.name.lower()
            ):
                result[role.name] = "name"
        return result

    @staticmethod
    def _drop_constant_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        to_drop = [name for name in df.columns if df[name].drop_nulls().n_unique() <= 1]
        return (df.drop(to_drop), to_drop) if to_drop else (df, [])

    def analyze(self) -> dict[str, Any]:
        """Compatibility alias for a rich data-quality profile."""
        profile = profile_dataset(
            self.df, PipelineConfig(categorical_threshold=self.categorical_threshold)
        )
        report = profile.to_dict()
        report["cols"] = self.df.width
        report["dtypes"] = dict(Counter(str(dtype) for dtype in self.df.dtypes))
        report["plugins"] = {plugin.name: plugin.analyze(self.df) for plugin in self.plugins}
        return report

    def recommend(self) -> list[str]:
        return [
            recommendation
            for plugin in self.plugins
            for recommendation in plugin.recommend(plugin.analyze(self.df))
        ]

    def profile(self) -> dict[str, Any]:
        return self.analyze()

    def plan(self) -> dict[str, Any]:
        config = self._config(drop_identifiers=False)
        return DataDocPipeline(config).plan(self.df).to_dict()

    def _config(self, **overrides: Any) -> PipelineConfig:
        values: dict[str, Any] = {
            "categorical_threshold": self.categorical_threshold,
            "outlier_multiplier": self.outlier_multiplier,
            "drop_identifiers": True,
            "scaling": "standard" if self.scaling_ratio > 0 else "none",
        }
        values.update(overrides)
        return PipelineConfig(**values)

    def engineer(
        self,
        progress_callback=None,
        categorical_threshold: int | None = None,
        scaling_ratio: float | None = None,
        outlier_multiplier: float | None = None,
    ) -> pl.DataFrame:
        """Fit and apply a legacy pipeline on the supplied dataset.

        This method remains for compatibility; use DataDocPipeline.fit() on a
        training split for leakage-safe model work.
        """
        warnings.warn(
            "DATADOC.engineer() is deprecated for supervised ML. Use DataDocPipeline.fit()/transform() with a training split.",
            DeprecationWarning,
            stacklevel=2,
        )
        if categorical_threshold is not None:
            self.categorical_threshold = categorical_threshold
        if scaling_ratio is not None:
            self.scaling_ratio = scaling_ratio
        if outlier_multiplier is not None:
            self.outlier_multiplier = outlier_multiplier
        self.plugins = [
            MissingValuePlugin(),
            OutlierPlugin(self.outlier_multiplier),
            DatetimePlugin(),
            CategoricalEncoderPlugin(self.categorical_threshold),
            ScalingPlugin(self.scaling_ratio),
        ]

        self._applied_plugins, self._skipped_plugins, self._dropped_columns = [], [], []
        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            should_apply = any(
                bool(value) for key, value in analysis.items() if key.startswith("has_")
            )
            if progress_callback:
                progress_callback(plugin.name, "running", [])
            if should_apply:
                self._applied_plugins.append(plugin.name)
                if progress_callback:
                    progress_callback(plugin.name, "applied", plugin.recommend(analysis))
            else:
                self._skipped_plugins.append(plugin.name)
                if progress_callback:
                    progress_callback(plugin.name, "skipped", [])

        self.fitted_pipeline = DataDocPipeline(self._config()).fit(self.df)
        self._dropped_columns = list(self.fitted_pipeline.state_["dropped"])
        return self.fitted_pipeline.transform(self.df)

    def compare(self, clean_df: pl.DataFrame) -> dict[str, Any]:
        original_missing = sum(self.df[name].null_count() for name in self.df.columns)
        clean_missing = sum(clean_df[name].null_count() for name in clean_df.columns)
        return {
            "original_shape": self.df.shape,
            "clean_shape": clean_df.shape,
            "rows_changed": self.df.height != clean_df.height,
            "cols_added": clean_df.width - self.df.width,
            "original_missing": original_missing,
            "clean_missing": clean_missing,
            "original_dtypes": dict(Counter(str(dtype) for dtype in self.df.dtypes)),
            "clean_dtypes": dict(Counter(str(dtype) for dtype in clean_df.dtypes)),
        }

    def list_plugins(self) -> list[dict[str, Any]]:
        result = []
        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            result.append(
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "priority": plugin.priority,
                    "will_trigger": any(
                        bool(value) for key, value in analysis.items() if key.startswith("has_")
                    ),
                    "dependencies": plugin.dependencies,
                }
            )
        return result

    def revert(self) -> None:
        self.df = self._original_df.clone()

    def apply_plugin_by_name(self, plugin_name: str) -> str:
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                self.df = plugin.apply(self.df)
                return f"Successfully applied {plugin_name}."
        return f"Error: Plugin {plugin_name} not found."

    def _extract_metadata(self) -> str:
        profile = profile_dataset(
            self.df, PipelineConfig(categorical_threshold=self.categorical_threshold)
        )
        return json.dumps(
            {
                "rows": profile.rows,
                "columns": profile.columns,
                "schema": profile.schema,
                "null_counts": profile.null_counts,
                "findings": profile.findings,
                "available_plugins": [plugin.name for plugin in self.plugins],
            },
            indent=2,
        )

    def _generate_ai_plan(
        self, model: str, goal: str, api_key: str | None = None
    ) -> list[dict[str, str]]:
        try:
            import litellm
        except ImportError as error:
            raise DataDocError(
                "AI planning requires the optional AI dependencies: pip install 'datadoc-cli[ai]'."
            ) from error
        prompt = (
            "Choose only from the available plugin names and return JSON shaped as "
            '{"plan": [{"plugin_name": str, "reason": str}]}. '
            f"Dataset metadata: {self._extract_metadata()}\nGoal: {goal}"
        )
        response = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}], api_key=api_key
        )
        data = json.loads(response.choices[0].message.content)
        plan = data.get("plan", [])
        allowed = {plugin.name for plugin in self.plugins}
        invalid = [
            item.get("plugin_name") for item in plan if item.get("plugin_name") not in allowed
        ]
        if invalid:
            raise DataDocError(
                f"AI plan contains unregistered plugins: {', '.join(map(str, invalid))}"
            )
        return [{"plugin": item["plugin_name"], "reason": item.get("reason", "")} for item in plan]

    def ai_analyze(self, model: str, api_key: str | None = None) -> str:
        try:
            import litellm
        except ImportError as error:
            raise DataDocError(
                "AI analysis requires the optional AI dependencies: pip install 'datadoc-cli[ai]'."
            ) from error
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this dataset profile without requesting raw rows:\n{self._extract_metadata()}",
                }
            ],
            api_key=api_key,
        )
        return response.choices[0].message.content.strip()

    def ai_recommend(
        self, model: str, goal: str, api_key: str | None = None
    ) -> list[dict[str, str]]:
        return self._generate_ai_plan(model, goal, api_key)

    def ai_engineer(self, model: str, goal: str, api_key: str | None = None) -> pl.DataFrame:
        """Safely execute only registered transformations selected by an AI plan."""
        plan = self.ai_recommend(model, goal, api_key)
        requested = {item["plugin"] for item in plan}
        self.revert()
        self._applied_plugins = []
        self._skipped_plugins = []
        roles = self._detect_column_roles(self.df)
        drop_columns = [name for name, role in roles.items() if role in {"id", "name"}]
        if drop_columns:
            self.df = self.df.drop(drop_columns)
        for plugin in self.plugins:
            if plugin.name in requested:
                self.df = plugin.apply(self.df)
                self._applied_plugins.append(plugin.name)
            else:
                self._skipped_plugins.append(plugin.name)
        return self.df


    def pipeline(self) -> str:
        if self.fitted_pipeline is None:
            self.engineer()
        assert self.fitted_pipeline is not None
        artifact = json.dumps(self.fitted_pipeline.to_dict(), indent=2)
        return f"""import json
import polars as pl
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig, read_dataset

ARTIFACT = json.loads({artifact!r})

def load_and_clean_data(file_path: str) -> pl.DataFrame:
    pipeline = DataDocPipeline(PipelineConfig(**ARTIFACT["config"]))
    pipeline.input_schema_ = ARTIFACT["input_schema"]
    pipeline.output_schema_ = ARTIFACT["output_schema"]
    pipeline.state_ = ARTIFACT["state"]
    pipeline.fitted_ = True
    return pipeline.transform(read_dataset(file_path))
"""

