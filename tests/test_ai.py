"""Unit tests for DATADOC AI feature engineering explainer and LLM integration."""

from unittest.mock import patch, MagicMock
import polars as pl
from typer.testing import CliRunner

from datadoc.ai.client import detect_provider_and_model, LLMClient
from datadoc.ai.prompts import build_profile_digest, build_analysis_prompt
from datadoc.ai.explainer import (
    AIExplainer,
    _parse_sections,
    _infer_recommended_preset,
    explain_dataset,
)
from datadoc.cli.app import app
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig

runner = CliRunner()


def test_detect_provider_and_model():
    p, m = detect_provider_and_model("gemini/gemini-2.0-flash")
    assert p == "google"
    assert m == "gemini/gemini-2.0-flash"

    p, m = detect_provider_and_model("claude-3-5-sonnet")
    assert p == "anthropic"

    p, m = detect_provider_and_model("ollama/llama3")
    assert p == "ollama"


def test_build_profile_digest(tmp_path):
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [25.0, 30.0, None, 45.0],
            "city": ["New York", "London", "London", "Tokyo"],
            "target": [0, 1, 0, 1],
        }
    )
    pipe = DataDocPipeline(PipelineConfig(target="target"))
    profile_dict = pipe.profile(df).to_dict()

    digest = build_profile_digest(profile_dict, df, target="target")
    assert "DATASET SUMMARY: 4 rows, 4 columns." in digest
    assert "Column 'age'" in digest
    assert "Column 'city'" in digest
    assert "PREDICTIVE TARGET COLUMN: 'target'" in digest

    prompt = build_analysis_prompt(digest, target="target")
    assert "Semantic Column Profiling" in prompt
    assert "Missingness Mechanisms" in prompt
    assert "Domain Feature Engineering Hypotheses" in prompt


def test_parse_sections_and_preset():
    sample_md = """### 1. Semantic Column Profiling & Hidden Sentinels
- Column id is an identifier.

### 2. Missingness Mechanisms & Target Leakage Risks
- Age is missing at random.

### 3. Domain Feature Engineering Hypotheses
- Interaction: `pl.col('A') / pl.col('B')`

### 4. Recommended DATADOC Transformation Strategy
- We strongly recommend the `tree` preset with Target Encoding.
"""
    sections = _parse_sections(sample_md)
    assert "semantics" in sections
    assert "missingness_and_leakage" in sections
    assert "feature_hypotheses" in sections
    assert "strategy" in sections

    preset = _infer_recommended_preset(sections["strategy"])
    assert preset == "tree"


def test_heuristic_fallback_on_demo():
    df = pl.read_csv("demo.csv")
    result = explain_dataset(df, target="churn")

    assert result.model == "heuristic-baseline"
    assert result.provider == "local-rules"
    assert "semantics" in result.sections
    assert "feature_hypotheses" in result.sections
    assert len(result.recommended_preset) > 0


def test_ai_explainer_with_mocked_llm():
    df = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "clicks": [10, 20, 30],
            "spend": [100.0, 200.0, 300.0],
            "converted": [0, 1, 1],
        }
    )

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""### 1. Semantic Column Profiling
- user_id is a primary key.

### 2. Missingness Mechanisms
- Zero missingness.

### 3. Domain Feature Engineering Hypotheses
- Formula: `pl.col('spend') / pl.col('clicks')` for cost-per-click.

### 4. Recommended Strategy
- Use `linear` preset with StandardScaler.
"""
            )
        )
    ]

    with patch("litellm.completion", return_value=mock_response):
        client = LLMClient(model="gpt-4o-mini")
        res = client.complete("test prompt")
        assert "Semantic Column Profiling" in res

        explainer = AIExplainer(model="gpt-4o-mini")
        result = explainer.explain(df, target="converted", model="gpt-4o-mini")
        assert result.recommended_preset == "linear"
        assert "feature_hypotheses" in result.sections


def test_cli_explain_command():
    res = runner.invoke(app, ["explain", "demo.csv", "--target", "churn"])
    assert res.exit_code == 0
    assert "DATADOC AI Dataset Explainer" in res.stdout
    assert "Semantic Column Profiling" in res.stdout


def test_agent_action_extraction_and_formatting():
    from datadoc.ai.agent_runner import DataDocAgent, _format_action_details

    agent = DataDocAgent.__new__(DataDocAgent)
    agent.target = "Survived"

    # Test JSON array parsing
    raw_json = '[{"type": "AddInteraction", "col_a": "Age", "col_b": "Fare", "op": "div", "rationale": "Age fare ratio"}]'
    actions = agent._extract_json_array(raw_json)
    assert len(actions) == 1
    assert actions[0]["col_a"] == "Age"
    assert _format_action_details(actions[0]) == "Age ÷ Fare"

    # Test markdown fenced JSON
    fenced_json = '```json\n[{"type": "ApplyTransform", "col": "Fare", "transform": "log1p", "rationale": "Log fare"}]\n```'
    actions = agent._extract_json_array(fenced_json)
    assert len(actions) == 1
    assert actions[0]["transform"] == "log1p"
    assert _format_action_details(actions[0]) == "log1p(Fare)"

    # Test action application to PipelineConfig
    cfg = PipelineConfig(target="Survived")
    agent._apply_actions(
        cfg,
        [
            {"type": "AddInteraction", "col_a": "Age", "col_b": "Fare", "op": "div"},
            {"type": "ApplyTransform", "col": "Fare", "transform": "log1p"},
            {"type": "SetScaling", "scaling": "robust"},
            {"type": "ToggleFeature", "setting": "clip_outliers", "value": True},
            {"type": "IgnoreColumn", "col": "PassengerId"},
        ],
    )
    assert len(cfg.custom_interactions) == 1
    assert len(cfg.custom_transforms) == 1
    assert cfg.scaling == "robust"
    assert cfg.clip_outliers is True
    assert "PassengerId" in cfg.ignored_columns

