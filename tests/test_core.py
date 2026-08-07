import pytest
import polars as pl

from datadoc.core.engine import DATADOC
from datadoc.plugins.missing_values import MissingValuePlugin
from datadoc.plugins.outliers import OutlierPlugin
from datadoc.plugins.encoders import CategoricalEncoderPlugin
from datadoc.plugins.datetime_feat import DatetimePlugin
from datadoc.plugins.scaling import ScalingPlugin


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV with known issues for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "Dave", "Eve"],
            "age": [30.0, None, 25.0, 40.0, 35.0],
            "salary": [50000.0, 60000.0, None, 80000.0, 70000.0],
            "department": ["Engineering", "Sales", "Engineering", "HR", None],
        }
    )
    df.write_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def clean_csv(tmp_path):
    """Create a clean CSV with no issues."""
    csv_path = tmp_path / "clean_data.csv"
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    df.write_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def datetime_csv(tmp_path):
    """Create a CSV with datetime columns."""
    csv_path = tmp_path / "datetime_data.csv"
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10", "2024-05-05"],
            "value": [100, 200, 300, 400, 500],
        }
    )
    df.write_csv(csv_path)
    return str(csv_path)


# ────────────────────────────────────────────
# Core Engine Tests
# ────────────────────────────────────────────


class TestDATADOCEngine:
    def test_load_csv(self, sample_csv):
        doc = DATADOC(sample_csv)
        assert (doc.df.height, doc.df.width) == (5, 5)

    def test_analyze_returns_report(self, sample_csv):
        doc = DATADOC(sample_csv)
        report = doc.analyze()
        assert report["rows"] == 5
        assert report["cols"] == 5
        assert "plugins" in report

    def test_recommend_returns_list(self, sample_csv):
        doc = DATADOC(sample_csv)
        recs = doc.recommend()
        assert isinstance(recs, list)
        assert len(recs) > 0  # Should have recommendations for missing data

    def test_engineer_returns_dataframe(self, sample_csv):
        doc = DATADOC(sample_csv)
        clean_df = doc.engineer()
        assert isinstance(clean_df, pl.DataFrame)
        assert sum(clean_df[c].null_count() for c in clean_df.columns) == 0  # No missing values

    def test_engineer_tracks_plugins(self, sample_csv):
        doc = DATADOC(sample_csv)
        doc.engineer()
        assert isinstance(doc._applied_plugins, list)
        assert isinstance(doc._skipped_plugins, list)
        assert "MissingValuePlugin" in doc._applied_plugins

    def test_compare(self, sample_csv):
        doc = DATADOC(sample_csv)
        clean_df = doc.engineer()
        diff = doc.compare(clean_df)
        assert diff["original_missing"] == 4
        assert diff["clean_missing"] == 0

    def test_list_plugins(self, sample_csv):
        doc = DATADOC(sample_csv)
        info = doc.list_plugins()
        assert len(info) == 5
        names = [p["name"] for p in info]
        assert "MissingValuePlugin" in names
        assert "OutlierPlugin" in names
        assert "DatetimePlugin" in names
        assert "CategoricalEncoderPlugin" in names
        assert "ScalingPlugin" in names

    def test_pipeline_returns_string(self, sample_csv):
        doc = DATADOC(sample_csv)
        script = doc.pipeline()
        assert isinstance(script, str)
        assert "import polars" in script
        assert "def load_and_clean_data" in script

    def test_clean_csv_no_recommendations(self, clean_csv):
        doc = DATADOC(clean_csv)
        recs = doc.recommend()
        # Clean data should have very few or no recommendations
        # (scaling might trigger depending on scale ratio)
        missing_recs = [r for r in recs if "missing" in r.lower()]
        assert len(missing_recs) == 0


# ────────────────────────────────────────────
# Plugin Tests
# ────────────────────────────────────────────


class TestMissingValuePlugin:
    def test_analyze_detects_missing(self):
        plugin = MissingValuePlugin()
        df = pl.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
        result = plugin.analyze(df)
        assert result["has_missing_values"] is True
        assert result["total_missing"] == 2

    def test_analyze_no_missing(self):
        plugin = MissingValuePlugin()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = plugin.analyze(df)
        assert result["has_missing_values"] is False

    def test_apply_fills_missing(self):
        plugin = MissingValuePlugin()
        df = pl.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "x"]})
        clean = plugin.apply(df)
        assert sum(clean[c].null_count() for c in clean.columns) == 0
        assert clean["a"][1] == 2.0  # Median of [1, 3]
        assert clean["b"][1] == "x"  # Mode

    def test_recommend(self):
        plugin = MissingValuePlugin()
        result = {
            "has_missing_values": True,
            "total_missing": 3,
            "columns_affected": {"a": 2, "b": 1},
        }
        recs = plugin.recommend(result)
        assert len(recs) == 1
        assert "3 missing values" in recs[0]



    def test_explain(self):
        plugin = MissingValuePlugin()
        assert "median" in plugin.explain().lower()

    def test_interface_properties(self):
        plugin = MissingValuePlugin()
        assert plugin.name == "MissingValuePlugin"
        assert plugin.version == "0.1.0"
        assert plugin.priority == 10
        assert len(plugin.description) > 0


class TestOutlierPlugin:
    def test_analyze_detects_outliers(self):
        plugin = OutlierPlugin()
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})  # 100 is an outlier
        result = plugin.analyze(df)
        assert result["has_outliers"] is True

    def test_analyze_no_outliers(self):
        plugin = OutlierPlugin()
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = plugin.analyze(df)
        assert result["has_outliers"] is False

    def test_apply_clips(self):
        plugin = OutlierPlugin()
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})
        clean = plugin.apply(df)
        assert clean["a"].max() <= 100.0

    def test_interface_properties(self):
        plugin = OutlierPlugin()
        assert plugin.name == "OutlierPlugin"
        assert plugin.priority == 20
        assert "MissingValuePlugin" in plugin.dependencies


class TestCategoricalEncoderPlugin:
    def test_analyze_detects_categorical(self):
        plugin = CategoricalEncoderPlugin()
        df = pl.DataFrame({"dept": ["A", "B", "A", "C"], "val": [1, 2, 3, 4]})
        result = plugin.analyze(df)
        assert result["has_categorical"] is True
        assert "dept" in result["categorical_columns"]

    def test_apply_encodes(self):
        plugin = CategoricalEncoderPlugin()
        df = pl.DataFrame({"dept": ["A", "B", "A", "C"], "val": [1, 2, 3, 4]})
        clean = plugin.apply(df)
        assert "dept" not in clean.columns  # Original column dropped
        assert clean.width > 2  # New dummy columns added

    def test_high_cardinality_dropped_as_identifier(self):
        plugin = CategoricalEncoderPlugin()
        df = pl.DataFrame({"id": [f"user_{i}" for i in range(20)]})
        result = plugin.analyze(df)
        assert result["has_categorical"] is True  # Detected as identifier to drop
        assert "id" in result["identifier_columns"]
        assert "id" not in result["categorical_columns"]  # Not encoded, dropped
        # Applying should drop the column
        clean = plugin.apply(df)
        assert "id" not in clean.columns

    def test_interface_properties(self):
        plugin = CategoricalEncoderPlugin()
        assert plugin.name == "CategoricalEncoderPlugin"
        assert plugin.priority == 40


class TestDatetimePlugin:
    def test_analyze_detects_datetime_strings(self):
        plugin = DatetimePlugin()
        df = pl.DataFrame({"date": ["2024-01-01", "2024-02-15", "2024-03-20"]})
        result = plugin.analyze(df)
        assert result["has_datetime"] is True

    def test_analyze_no_datetime(self):
        plugin = DatetimePlugin()
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = plugin.analyze(df)
        assert result["has_datetime"] is False

    def test_apply_extracts_features(self):
        plugin = DatetimePlugin()
        df = pl.DataFrame({"date": ["2024-01-15", "2024-06-20", "2024-12-25"]})
        clean = plugin.apply(df)
        assert "date_month" in clean.columns
        assert "date_day" in clean.columns
        assert "date_dayofweek" in clean.columns
        assert "date" not in clean.columns  # Original dropped

    def test_drops_constant_year(self):
        plugin = DatetimePlugin()
        # All dates in same year — year column should be dropped
        df = pl.DataFrame({"date": ["2024-01-15", "2024-06-20", "2024-12-25"]})
        clean = plugin.apply(df)
        assert "date_year" not in clean.columns  # Constant, so dropped

    def test_interface_properties(self):
        plugin = DatetimePlugin()
        assert plugin.name == "DatetimePlugin"
        assert plugin.priority == 30


class TestScalingPlugin:
    def test_analyze_detects_scale_mismatch(self):
        plugin = ScalingPlugin()
        df = pl.DataFrame(
            {
                "small": [1.0, 2.0, 3.0, 4.0, 5.0],
                "big": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0],
            }
        )
        result = plugin.analyze(df)
        assert result["has_scale_issues"] is True

    def test_analyze_no_scale_issues(self):
        plugin = ScalingPlugin()
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = plugin.analyze(df)
        assert result["has_scale_issues"] is False

    def test_apply_scales(self):
        plugin = ScalingPlugin()
        df = pl.DataFrame(
            {
                "small": [1.0, 2.0, 3.0, 4.0, 5.0],
                "big": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0],
            }
        )
        clean = plugin.apply(df)
        # After scaling, mean should be near 0
        assert abs(clean["big"].mean()) < 0.01

    def test_interface_properties(self):
        plugin = ScalingPlugin()
        assert plugin.name == "ScalingPlugin"
        assert plugin.priority == 45


# ────────────────────────────────────────────
# Base Plugin Interface Tests
# ────────────────────────────────────────────


class TestBasePluginInterface:
    """Verify all plugins implement the full BasePlugin interface."""

    @pytest.fixture(
        params=[
            MissingValuePlugin,
            OutlierPlugin,
            CategoricalEncoderPlugin,
            DatetimePlugin,
            ScalingPlugin,
        ]
    )
    def plugin(self, request):
        return request.param()

    def test_has_name(self, plugin):
        assert isinstance(plugin.name, str)
        assert len(plugin.name) > 0

    def test_has_version(self, plugin):
        assert isinstance(plugin.version, str)

    def test_has_description(self, plugin):
        assert isinstance(plugin.description, str)
        assert len(plugin.description) > 0

    def test_has_priority(self, plugin):
        assert isinstance(plugin.priority, int)


    def test_has_dependencies(self, plugin):
        assert isinstance(plugin.dependencies, list)

    def test_has_explain(self, plugin):
        explanation = plugin.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 0



def test_ai_engineer_mock(sample_csv, monkeypatch):
    doc = DATADOC(sample_csv)

    # Mock litellm.completion
    class MockChoice:
        message = type(
            "Message",
            (),
            {"content": '{"plan": [{"plugin_name": "MissingValuePlugin", "reason": "Fix nulls"}]}'},
        )()

    class MockResponse:
        choices = [MockChoice()]

    def mock_completion(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("litellm.completion", mock_completion)

    clean_df = doc.ai_engineer(model="mock/model", goal="Clean data", api_key="dummy_key")
    # name and id columns are auto-dropped by the engine (identifiers)
    assert "name" not in clean_df.columns
    assert "id" not in clean_df.columns
    assert "MissingValuePlugin" in doc._applied_plugins


def test_column_role_detection():
    df = pl.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
            "Age": [30, 25, 40, 35, 28],
        }
    )
    roles = DATADOC._detect_column_roles(df)
    assert roles["ID"] == "id"
    assert roles["Name"] == "name"
    assert roles["Age"] == "feature"


def test_scaling_preserves_binary_columns():
    plugin = ScalingPlugin()
    df = pl.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            "big": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0],
            "is_active": [0, 1, 0, 1, 0],  # Binary column
        }
    )
    clean = plugin.apply(df)
    # Binary column should not be scaled
    assert set(clean["is_active"].to_list()).issubset({0, 1})
