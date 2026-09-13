import polars as pl
from datadoc.plugins.base import BasePlugin


class DuplicateRemoverPlugin(BasePlugin):
    """Detect exact duplicate rows. Stateless advisor; fitted pipeline dedups on train when enabled."""

    @property
    def name(self) -> str:
        return "DuplicateRemoverPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects exact duplicate rows; optionally drops them during fit (deduplicate=True)."

    @property
    def priority(self) -> int:
        return 5  # First — before imputation so stats aren't skewed

    def analyze(self, df: pl.DataFrame) -> dict:
        dup = int(df.is_duplicated().sum()) if df.height else 0
        return {
            "has_duplicates": dup > 0,
            "duplicate_rows": dup,
            "duplicate_ratio": (dup / df.height) if df.height else 0.0,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        if analysis_result.get("has_duplicates"):
            n = analysis_result["duplicate_rows"]
            r = analysis_result.get("duplicate_ratio", 0.0)
            return [
                f"Found {n} duplicate rows ({r:.1%}). "
                "Recommendation: enable deduplicate=True to drop them at fit time (train-only)."
            ]
        return []

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        # Stateless demo: drop duplicates. Production path uses PipelineConfig.deduplicate.
        return df.unique(maintain_order=True)

    def explain(self) -> str:
        return (
            "DuplicateRemoverPlugin counts exact duplicate rows via is_duplicated(). "
            "With deduplicate=True, DataDocPipeline drops them from training data at fit "
            "so medians/vocabularies aren't skewed."
        )
