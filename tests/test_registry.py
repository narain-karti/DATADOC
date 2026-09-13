import polars as pl

from datadoc.core.pipeline import DataDocPipeline, PipelineConfig
from datadoc.plugins.registry import BUILTIN_PLUGIN_NAMES, get_plugin, list_plugins
from datadoc.plugins.rare import RareCategoryPlugin
from datadoc.plugins.duplicates import DuplicateRemoverPlugin


def test_registry_lists_nine_plugins_sorted():
    plugs = list_plugins()
    names = [p.name for p in plugs]
    assert len(plugs) == 9
    for expected in BUILTIN_PLUGIN_NAMES:
        assert expected in names
    priorities = [p.priority for p in plugs]
    assert priorities == sorted(priorities)
    assert names[0] == "DuplicateRemoverPlugin"


def test_get_plugin_case_insensitive():
    assert get_plugin("rareCategoryPlugin").name == "RareCategoryPlugin"
    assert get_plugin("missingvalueplugin").name == "MissingValuePlugin"
    assert get_plugin("nope") is None


def test_duplicate_plugin_detect_and_apply():
    df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    p = DuplicateRemoverPlugin()
    assert p.analyze(df)["duplicate_rows"] == 2
    assert p.apply(df).height == 2


def test_deduplicate_config_drops_train_duplicates():
    df = pl.DataFrame({"a": [1, 1, 2, 3], "t": [0, 0, 1, 0]})
    pipe = DataDocPipeline(PipelineConfig(target="t", scaling="none", deduplicate=True)).fit(df)
    assert pipe.train_provenance_["deduplicated_rows"] == 1
    assert any(op["operation"] == "deduplicate" for op in pipe.plan_.operations)


def test_rare_grouping_caps_one_hot():
    cats = ["common"] * 80 + [f"rare_{i}" for i in range(20)]
    df = pl.DataFrame({"c": cats, "t": [0] * 100})
    pipe = DataDocPipeline(
        PipelineConfig(target="t", scaling="none", rare_category_min_frequency=0.05)
    ).fit(df)
    spec = pipe.state_["categorical"]["c"]
    assert spec["kind"] == "one_hot"
    assert "__RARE__" in spec["categories"]
    out = pipe.transform(pl.DataFrame({"c": ["rare_0", "common", "unseen_xyz"]}))
    assert "c____RARE__" in out.columns
    assert out["c____RARE__"].to_list()[0] == 1
    assert out["c__common"].to_list()[1] == 1


def test_rare_plugin_advisor():
    df = pl.DataFrame({"c": ["a"] * 90 + ["b"] * 5 + ["c"] * 5})
    p = RareCategoryPlugin(min_frequency=0.06)
    res = p.analyze(df)
    assert res["has_rare"] is True
    assert "c" in res["rare_columns"]
    assert p.recommend(res)


def test_datetime_hour_and_cyclical():
    df = pl.DataFrame(
        {
            "created_at": ["2024-01-01 10:00:00", "2024-02-01 12:30:00", "2024-03-01 08:00:00"],
            "t": [0, 1, 0],
        }
    )
    pipe = DataDocPipeline(PipelineConfig(target="t", scaling="none")).fit(df)
    assert pipe.state_["datetime"]["created_at"]["has_hour"] is True
    assert "created_at__hour" in pipe.transform(df).columns

    pipe2 = DataDocPipeline(PipelineConfig(target="t", scaling="none", datetime_cyclical=True)).fit(
        df
    )
    out = pipe2.transform(df)
    assert "created_at__month__sin" in out.columns
    assert "created_at__month__cos" in out.columns


def test_provenance_and_v1_backcompat(tmp_path):
    import json

    df = pl.DataFrame({"a": [1, 2, 3], "t": [0, 1, 0]})
    pipe = DataDocPipeline(PipelineConfig(target="t", scaling="none")).fit(df)
    d = pipe.to_dict()
    assert d["artifact_version"] == 2
    assert d["provenance"]["rows"] == 3
    path = tmp_path / "p.json"
    pipe.save(path)
    assert DataDocPipeline.load(path).transform(df).equals(pipe.transform(df))

    # Simulate v1 artifact (no provenance, no new datetime keys)
    v1 = dict(d)
    v1["artifact_version"] = 1
    v1.pop("provenance", None)
    for spec in v1["state"]["datetime"].values():
        spec.pop("has_hour", None)
        spec.pop("cyclical", None)
    p1 = tmp_path / "v1.json"
    p1.write_text(json.dumps(v1), encoding="utf-8")
    loaded = DataDocPipeline.load(p1)
    assert loaded.transform(df).height == 3


def test_both_estimator_evaluates_both_families():
    import pytest

    pytest.importorskip("sklearn")
    import polars as pl

    values = list(range(60))
    df = pl.DataFrame(
        {
            "amount": values,
            "category": ["high" if value % 2 else "low" for value in values],
            "target": [1 if value % 2 else 0 for value in values],
        }
    )
    report = DataDocPipeline(
        PipelineConfig(target="target", task="classification", estimator_family="both")
    ).evaluate(df)
    assert report.estimator_family == "both"
    assert 0.0 <= report.baseline_score <= 1.0
    assert 0.0 <= report.selected_score <= 1.0
    # "both" must resolve scaling like linear (trees are scale-invariant either way)
    assert PipelineConfig(estimator_family="both").resolved_scaling() == "standard"


def test_drift_and_ablation_helpers():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "t": [0, 1, 0, 1]})
    pipe = DataDocPipeline(PipelineConfig(target="t", scaling="none")).fit(df)
    drift = pipe.drift_report(df)
    assert drift["schema_ok"] is True
    assert "issues" in drift
    assert "provenance" in drift


def test_explain_plan_and_html():
    df = pl.DataFrame({"a": [1, 2, 3], "t": [0, 1, 0]})
    pipe = DataDocPipeline(PipelineConfig(target="t", scaling="none"))
    text = pipe.explain_plan(df)
    assert "numeric_imputation" in text
    html = pipe.profile_to_html(df)
    assert "DATADOC Profile" in html
    assert "<table" in html
