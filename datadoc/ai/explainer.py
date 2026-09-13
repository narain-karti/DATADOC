"""AI Dataset Explainer and Feature Engineering engine for DATADOC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import polars as pl

from datadoc.ai.client import get_llm_client, is_ai_configured, detect_provider_and_model
from datadoc.ai.prompts import SYSTEM_PROMPT, build_analysis_prompt, build_profile_digest
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig


@dataclass
class AIExplanationResult:
    """Structured response containing AI feature engineering insights and recommendations."""

    model: str
    provider: str
    raw_markdown: str
    sections: dict[str, str] = field(default_factory=dict)
    recommended_preset: str = "balanced"
    recommended_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "recommended_preset": self.recommended_preset,
            "sections": self.sections,
            "recommended_config": self.recommended_config,
            "raw_markdown": self.raw_markdown,
        }


def _parse_sections(markdown_text: str) -> dict[str, str]:
    """Splits the markdown into standard named sections."""
    sections: dict[str, str] = {}
    current_key = "overview"
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        lower_line = line.lower()
        if "semantic" in lower_line and ("profiling" in lower_line or "role" in lower_line):
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            current_key = "semantics"
        elif "missingness" in lower_line or "leakage" in lower_line:
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            current_key = "missingness_and_leakage"
        elif "feature engineering" in lower_line or "hypotheses" in lower_line:
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            current_key = "feature_hypotheses"
        elif "recommended" in lower_line and ("strategy" in lower_line or "preset" in lower_line):
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            current_key = "strategy"
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def _infer_recommended_preset(text: str) -> str:
    """Detects recommended preset from LLM strategy section."""
    lower = text.lower()
    if "tree" in lower:
        return "tree"
    if "linear" in lower:
        return "linear"
    if "robust" in lower:
        return "robust"
    return "balanced"


def _generate_heuristic_explanation(
    df: pl.DataFrame, profile_dict: dict[str, Any], target: Optional[str]
) -> AIExplanationResult:
    """Generates an intelligent heuristic baseline explanation when no LLM API key is present."""
    null_counts = profile_dict.get("null_counts", {})
    findings = profile_dict.get("findings", [])
    rows = df.height
    cols = df.width

    high_null_cols = [c for c, n in null_counts.items() if (n / rows) > 0.20] if rows else []
    id_cols = [f.get("column") for f in findings if f.get("code") == "identifier"]
    outlier_cols = [f.get("column") for f in findings if f.get("code") == "outliers"]

    num_cols = [c for c in df.columns if c != target and df[c].dtype.is_numeric()]
    cat_cols = [
        c
        for c in df.columns
        if c != target
        and (
            df[c].dtype in (pl.String, pl.Categorical)
            or "String" in str(df[c].dtype)
            or "Utf8" in str(df[c].dtype)
        )
    ]

    # Hypotheses generation heuristics
    hypotheses: list[str] = []
    if len(num_cols) >= 2:
        c1, c2 = num_cols[0], num_cols[1]
        hypotheses.append(
            f"- **Interaction Ratio (`{c1}_per_{c2}`)**: `pl.col('{c1}') / (pl.col('{c2}').abs() + 1.0)`\n"
            f"  *Rationale*: Normalizes `{c1}` relative to magnitude of `{c2}`, stabilizing variance for linear and tree models."
        )
    if high_null_cols:
        col = high_null_cols[0]
        hypotheses.append(
            f"- **Missingness Indicator (`{col}_is_missing`)**: `pl.col('{col}').is_null().cast(pl.Int8)`\n"
            f"  *Rationale*: Column `{col}` has >20% nulls. Missingness itself may carry systemic predictive signal (MNAR)."
        )
    if num_cols:
        c = num_cols[0]
        hypotheses.append(
            f"- **Log-Transform (`log1p_{c}`)**: `pl.col('{c}').log1p()`\n"
            f"  *Rationale*: Compresses positive right-skewed distribution, mitigating gradient instability in regression and neural networks."
        )

    preset = "tree" if len(cat_cols) > len(num_cols) else "balanced"

    md = f"""### 1. Semantic Column Profiling & Hidden Sentinels
- Dataset comprises **{rows:,} rows** and **{cols} features**.
- **Candidate Identifiers**: {", ".join(str(c) for c in id_cols) if id_cols else "None detected"}. Identifier columns carry no generalization power and should be dropped.
- **Continuous Features**: {len(num_cols)} columns ({", ".join(num_cols[:4])}{"..." if len(num_cols) > 4 else ""}).
- **Categorical Features**: {len(cat_cols)} columns ({", ".join(cat_cols[:4])}{"..." if len(cat_cols) > 4 else ""}).

### 2. Missingness Mechanisms & Target Leakage Risks
- **Missingness Audit**: {len(high_null_cols)} features exhibit significant missingness (>20% nulls): {", ".join(high_null_cols) if high_null_cols else "None"}.
- **Leakage Risks**: Verify that all features are observable at prediction time. Target column '{target or "unknown"}' must never be used as an input feature.

### 3. Domain Feature Engineering Hypotheses
{chr(10).join(hypotheses) if hypotheses else "- No immediate heuristic interaction candidates found."}

### 4. Recommended DATADOC Transformation Strategy
- **Recommended Preset**: `{preset}`
- **Encoding Strategy**: Use **Target Encoding** or One-Hot Encoding for low-cardinality features.
- **Outlier Capping**: Enable IQR capping (`clip_outliers = true`) for features with extreme tails: {", ".join(str(c) for c in outlier_cols[:3]) if outlier_cols else "Standard bounds"}.

*(Note: Heuristic baseline active. Set `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or use `--model ollama/llama3` for full LLM-driven deep domain reasoning.)*"""

    sections = _parse_sections(md)
    return AIExplanationResult(
        model="heuristic-baseline",
        provider="local-rules",
        raw_markdown=md,
        sections=sections,
        recommended_preset=preset,
        recommended_config={"task": "auto", "preset": preset, "clip_outliers": bool(outlier_cols)},
    )


class AIExplainer:
    """Orchestrates AI analysis and feature engineering recommendations for tabular datasets."""

    def __init__(self, model: Optional[str] = None, timeout: int = 45):
        self.preferred_model = model
        self.timeout = timeout

    def explain(
        self,
        df: pl.DataFrame,
        target: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AIExplanationResult:
        """Analyzes dataset profile with LLM or fallback heuristic."""
        chosen_model = model or self.preferred_model

        # Generate standard statistical profile
        pipe_cfg = PipelineConfig(target=target)
        pipeline = DataDocPipeline(pipe_cfg)
        profile_dict = pipeline.profile(df).to_dict()

        # Check if an API key or local model is available
        provider, resolved_model = detect_provider_and_model(chosen_model)
        if (
            provider == "none"
            and not is_ai_configured()
            and not (chosen_model and "ollama" in chosen_model)
        ):
            return _generate_heuristic_explanation(df, profile_dict, target)

        # Build prompt digest and execute LLM completion
        digest = build_profile_digest(profile_dict, df, target)
        user_prompt = build_analysis_prompt(digest, target)
        client = get_llm_client(model=chosen_model, timeout=self.timeout)

        raw_text = client.complete(user_prompt, system_prompt=SYSTEM_PROMPT)
        sections = _parse_sections(raw_text)
        preset = _infer_recommended_preset(sections.get("strategy", raw_text))

        config = {
            "task": "auto",
            "preset": preset,
            "drop_identifiers": True,
            "clip_outliers": "outlier" in raw_text.lower(),
        }

        return AIExplanationResult(
            model=client.model,
            provider=client.provider,
            raw_markdown=raw_text,
            sections=sections,
            recommended_preset=preset,
            recommended_config=config,
        )


def explain_dataset(
    df: pl.DataFrame, target: Optional[str] = None, model: Optional[str] = None
) -> AIExplanationResult:
    """Convenience helper to explain a dataset."""
    return AIExplainer(model=model).explain(df, target=target)
