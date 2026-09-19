import tempfile
from pathlib import Path
import polars as pl
import pytest

from datadoc.core.pipeline import DataDocPipeline, PipelineConfig
from datadoc.plugins.base import BasePlugin
from datadoc.plugins.registry import list_plugins, resolve_plugins
from datadoc.plugins.missing_values import MissingValuePlugin
from datadoc.plugins.scaling import ScalingPlugin
from datadoc.plugins.outliers import OutlierPlugin


class DummyCustomPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "DummyCustomPlugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Custom plugin that creates an engineered flag."

    @property
    def priority(self) -> int:
        return 99

    def analyze(self, df: pl.DataFrame) -> dict:
        return {"can_run": "val" in df.columns}

    def recommend(self, analysis_result: dict) -> list[str]:
        return ["Add flag column"] if analysis_result.get("can_run") else []

    def fit(self, df: pl.DataFrame) -> dict:
        # Learn threshold from training data only
        val_median = df["val"].median() if "val" in df.columns else 0.0
        return {"threshold": float(val_median or 0.0)}

    def transform(self, df: pl.DataFrame, state: dict | None = None) -> pl.DataFrame:
        state = state or {}
        threshold = state.get("threshold", 0.0)
        if "val" in df.columns:
            return df.with_columns(
                (pl.col("val") > threshold).cast(pl.UInt8).alias("val_above_train_median")
            )
        return df


def test_resolve_plugins():
    # Resolve by exact name
    plugs = resolve_plugins(["MissingValuePlugin", "ScalingPlugin"])
    assert len(plugs) == 2
    assert plugs[0].name == "MissingValuePlugin"  # priority 10 < 80

    # Resolve by partial/case-insensitive keyword
    plugs_partial = resolve_plugins(["scaling", "outlier"])
    assert len(plugs_partial) == 2
    names = [p.name for p in plugs_partial]
    assert "ScalingPlugin" in names
    assert "OutlierPlugin" in names

    # Resolve 'builtin'
    all_builtin = resolve_plugins(["builtin"])
    assert len(all_builtin) >= 9


def test_pipeline_with_plugin_lifecycle_and_serialization(tmp_path):
    # Test data
    train_with_nulls = pl.DataFrame({
        "num": [10.0, None, 30.0],
        "target": [0, 1, 0]
    })
    test_with_nulls = pl.DataFrame({
        "num": [None, 50.0],
        "target": [1, 0]
    })

    config = PipelineConfig(
        target="target",
        scaling="none",
        plugins=["MissingValuePlugin"]
    )
    pipeline = DataDocPipeline(config).fit(train_with_nulls)

    # Check plugin state was recorded in pipeline state_
    assert "plugins" in pipeline.state_
    assert "MissingValuePlugin" in pipeline.state_["plugins"]
    imputations = pipeline.state_["plugins"]["MissingValuePlugin"]["imputations"]
    # Train median of [10.0, 30.0] is 20.0
    assert imputations["num"] == 20.0

    # Transform test set: test null should be filled with train median (20.0), NOT test median
    transformed_test = pipeline.transform(test_with_nulls)
    assert transformed_test["num"].to_list()[0] == 20.0
    assert transformed_test["num"].null_count() == 0

    # Test artifact save and load
    art_path = tmp_path / "pipeline.json"
    pipeline.save(art_path)

    loaded_pipe = DataDocPipeline.load(art_path)
    loaded_trans = loaded_pipe.transform(test_with_nulls)
    assert loaded_trans.equals(transformed_test)


def test_plan_operations_includes_plugins():
    df = pl.DataFrame({
        "a": [1, 2, 3],
        "target": [0, 1, 0]
    })
    config = PipelineConfig(
        target="target",
        plugins=["ScalingPlugin", "OutlierPlugin"]
    )
    plan = DataDocPipeline(config).plan(df)
    ops = [op["operation"] for op in plan.operations]
    assert "plugin_OutlierPlugin" in ops
    assert "plugin_ScalingPlugin" in ops


def test_legacy_apply_backward_compatibility():
    class LegacyPlugin(BasePlugin):
        @property
        def name(self) -> str:
            return "LegacyPlugin"

        def analyze(self, df: pl.DataFrame) -> dict:
            return {}

        def recommend(self, analysis_result: dict) -> list[str]:
            return []

        def apply(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit(1).alias("legacy_flag"))

    lp = LegacyPlugin()
    df = pl.DataFrame({"x": [10, 20]})
    # Calling transform without fit should fallback to apply
    res = lp.transform(df)
    assert "legacy_flag" in res.columns
    assert res["legacy_flag"].to_list() == [1, 1]
