import polars as pl

from datadoc.core.compare import compare_datasets, compare_to_html
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig
from datadoc.core.report import generate_html_report
from datadoc.plugins.polynomial import PolynomialFeaturesPlugin
from datadoc.plugins.target_encoder import TargetEncoderPlugin


def test_target_encoder_plugin():
    df = pl.DataFrame(
        {
            "category": ["A", "A", "A", "B", "B", "C"],
            "target": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    plugin = TargetEncoderPlugin(target="target", smoothing=2.0)
    analysis = plugin.analyze(df)
    assert analysis["has_target"] is True
    assert "category" in analysis["candidate_columns"]

    out = plugin.apply(df)
    assert "category__target_enc" in out.columns
    # Global mean = (1+1+0+0+0+1)/6 = 0.5
    # Category A mean = 2/3 = 0.6667, count=3
    # Smoothed A = (3 * (2/3) + 2 * 0.5) / (3 + 2) = (2 + 1) / 5 = 0.6
    vals = out.filter(pl.col("category") == "A")["category__target_enc"].to_list()
    assert abs(vals[0] - 0.6) < 1e-4

    # Test explanation and recommendations
    assert plugin.recommend(analysis)
    assert "TargetEncoderPlugin" in plugin.explain()


def test_polynomial_features_plugin():
    df = pl.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "label": [0, 1, 0],
        }
    )
    plugin = PolynomialFeaturesPlugin(target="label", max_numeric_features=2)
    analysis = plugin.analyze(df)
    assert analysis["has_numeric"] is True
    assert analysis["pairs_count"] == 1
    assert analysis["squared_count"] == 2

    out = plugin.apply(df)
    assert "x1__pow2" in out.columns
    assert "x2__pow2" in out.columns
    assert "x1__x__x2" in out.columns
    assert out["x1__pow2"].to_list() == [1.0, 4.0, 9.0]
    assert out["x1__x__x2"].to_list() == [4.0, 10.0, 18.0]
    assert plugin.recommend(analysis)
    assert "PolynomialFeaturesPlugin" in plugin.explain()


def test_generate_html_report():
    df = pl.DataFrame(
        {
            "age": [22.0, 38.0, None, 35.0, 54.0],
            "fare": [7.25, 71.28, 8.05, 53.10, 8.05],
            "embarked": ["S", "C", "S", "S", "Q"],
            "survived": [0, 1, 1, 1, 0],
        }
    )
    html = generate_html_report(df, target="survived", title="Test Audit Report")
    assert "<!DOCTYPE html>" in html
    assert "Test Audit Report" in html
    assert "Health Score" in html
    assert "Schema &amp; Role" in html or "Schema & Role" in html
    assert "Recommended Preparation Pipeline Plan" in html


def test_compare_datasets():
    raw_df = pl.DataFrame(
        {
            "age": [20.0, 40.0, None, 30.0],
            "category": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )
    # Fit and transform with DATADOC
    pipe = DataDocPipeline(PipelineConfig(target="target", scaling="standard")).fit(raw_df)
    transformed_df = pipe.transform(raw_df)

    comp = compare_datasets(raw_df, transformed_df, target="target")
    assert comp.raw_rows == 4
    assert comp.transformed_rows == 4
    assert comp.raw_null_cells == 1
    assert comp.transformed_null_cells == 0
    assert comp.null_reduction == 1
    assert comp.null_reduction_pct == 100.0
    assert "age" in comp.retained_columns
    assert len(comp.engineered_columns) > 0

    # HTML comparison export
    html = compare_to_html(comp)
    assert "<!DOCTYPE html>" in html
    assert "Missing Values Resolution" in html
    assert "Column Lifecycle &amp; Lineage" in html or "Column Lifecycle & Lineage" in html
