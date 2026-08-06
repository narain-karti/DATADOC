import polars as pl
import pytest

from datadoc.core.pipeline import (
    DataDocError,
    DataDocPipeline,
    PipelineConfig,
    profile_dataset,
    read_dataset,
    write_dataset,
)


def test_numeric_statistics_are_fitted_only_on_training_data():
    train = pl.DataFrame({"age": [1.0, None, 3.0], "target": [0, 1, 0]})
    inference = pl.DataFrame({"age": [None]})
    pipeline = DataDocPipeline(PipelineConfig(target="target", scaling="none")).fit(train)

    transformed = pipeline.transform(inference)

    assert transformed["age"][0] == 2.0
    assert transformed["age__missing"][0] == 1


def test_unseen_one_hot_categories_are_handled_without_schema_drift():
    train = pl.DataFrame({"city": ["A", "B", "A"], "target": [0, 1, 0]})
    pipeline = DataDocPipeline(PipelineConfig(target="target", scaling="none")).fit(train)

    transformed = pipeline.transform(pl.DataFrame({"city": ["UNSEEN"]}))

    assert set(transformed.columns) == {"city__A", "city__B"}
    assert transformed.row(0) == (0, 0)


def test_identifier_detection_never_uses_numeric_shape_alone():
    df = pl.DataFrame({"age": [20, 25, 30, 35], "customer_id": [1, 2, 3, 4]})
    profile = profile_dataset(df)
    roles = {role.name: role.role for role in profile.roles}

    assert roles["age"] == "feature_numeric"
    assert roles["customer_id"] == "identifier"


def test_identifier_is_retained_unless_explicitly_enabled_for_drop():
    df = pl.DataFrame({"customer_id": [1, 2, 3], "amount": [1.0, 2.0, 3.0]})
    retained = DataDocPipeline(PipelineConfig(scaling="none")).fit_transform(df)
    dropped = DataDocPipeline(PipelineConfig(scaling="none", drop_identifiers=True)).fit_transform(
        df
    )

    assert "customer_id" in retained.columns
    assert "customer_id" not in dropped.columns


def test_all_null_columns_are_dropped_with_a_profile_finding():
    df = pl.DataFrame({"empty": [None, None], "value": [1, 2]})
    pipeline = DataDocPipeline(PipelineConfig(scaling="none")).fit(df)

    assert "empty" not in pipeline.transform(df).columns
    assert any(finding["code"] == "all_null" for finding in pipeline.profile_.findings)


def test_pipeline_artifact_round_trip_preserves_transformation(tmp_path):
    train = pl.DataFrame({"amount": [1.0, None, 3.0], "city": ["A", "B", "A"]})
    pipeline = DataDocPipeline(PipelineConfig(scaling="none")).fit(train)
    path = tmp_path / "pipeline.json"
    pipeline.save(path)

    expected = pipeline.transform(train)
    actual = DataDocPipeline.load(path).transform(train)

    assert actual.equals(expected)


def test_transform_rejects_missing_required_source_columns():
    pipeline = DataDocPipeline(PipelineConfig(scaling="none")).fit(
        pl.DataFrame({"amount": [1.0, 2.0]})
    )

    with pytest.raises(DataDocError, match="missing required columns"):
        pipeline.transform(pl.DataFrame({"other": [1.0]}))


def test_csv_and_parquet_round_trip(tmp_path):
    df = pl.DataFrame({"value": [1, 2, 3]})
    for suffix in (".csv", ".parquet"):
        path = tmp_path / f"data{suffix}"
        write_dataset(df, path)
        assert read_dataset(path).equals(df)


def test_evaluation_returns_a_reproducible_holdout_report():
    pytest.importorskip("sklearn")
    values = list(range(60))
    df = pl.DataFrame(
        {
            "amount": values,
            "category": ["high" if value % 2 else "low" for value in values],
            "target": [1 if value % 2 else 0 for value in values],
        }
    )

    report = DataDocPipeline(
        PipelineConfig(target="target", task="classification", scaling="standard")
    ).evaluate(df)

    assert report.metric == "balanced_accuracy"
    assert report.split_strategy == "stratified_holdout"
    assert 0.0 <= report.baseline_score <= 1.0
    assert 0.0 <= report.selected_score <= 1.0
