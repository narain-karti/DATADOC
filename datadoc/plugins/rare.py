import polars as pl
from datadoc.plugins.base import BasePlugin

RARE_TOKEN = "__RARE__"


class RareCategoryPlugin(BasePlugin):
    """Group infrequent categories into __RARE__ to cap one-hot explosion.

    Mirrors PipelineConfig.rare_category_min_frequency. Priority 42 runs right
    after CategoricalEncoderPlugin (40) conceptually, but in the fitted pipeline
    the grouping happens inside fit() before one-hot/frequency encoding.
    """

    def __init__(self, min_frequency: float = 0.02, max_categories: int = 20):
        self._min_frequency = min_frequency
        self._max_categories = max_categories
        super().__init__()

    @property
    def name(self) -> str:
        return "RareCategoryPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return (
            "Groups rare categories (< min_frequency) into __RARE__ to prevent one-hot explosion."
        )

    @property
    def priority(self) -> int:
        return 42

    def _rare_map(self, series: pl.Series) -> dict[str, list[str]]:
        values = series.cast(pl.String).fill_null("__MISSING__")
        counts = values.value_counts()
        total = max(values.len(), 1)
        freq = {str(r[0]): int(r[1]) / total for r in counts.iter_rows()}
        rare = sorted([k for k, f in freq.items() if f < self._min_frequency])
        kept = sorted([k for k, f in freq.items() if f >= self._min_frequency])
        return {"rare": rare, "kept": kept, "frequencies": freq}

    def analyze(self, df: pl.DataFrame) -> dict:
        str_cols = [c for c in df.columns if df[c].dtype == pl.String]
        rare_cols: dict[str, list[str]] = {}
        for col in str_cols:
            info = self._rare_map(df[col])
            # flag if rare exist AND cardinality would otherwise explode
            if info["rare"] and (
                len(info["kept"]) + len(info["rare"]) > self._max_categories or info["rare"]
            ):
                # Only report when there is at least one rare value
                rare_cols[col] = info["rare"]
        return {
            "has_rare": bool(rare_cols),
            "rare_columns": list(rare_cols.keys()),
            "rare_values": rare_cols,
            "min_frequency": self._min_frequency,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        if not analysis_result.get("has_rare"):
            return []
        recs = []
        for col, vals in analysis_result.get("rare_values", {}).items():
            preview = ", ".join(vals[:5]) + ("..." if len(vals) > 5 else "")
            recs.append(
                f"Column '{col}' has {len(vals)} rare categories (< {self._min_frequency:.1%}): {preview}. "
                f"Recommendation: group into {RARE_TOKEN} via rare_category_min_frequency={self._min_frequency}."
            )
        return recs

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        analysis = self.analyze(df_clean)
        for col in analysis.get("rare_columns", []):
            rare_set = set(analysis["rare_values"][col])
            df_clean = df_clean.with_columns(
                pl.when(pl.col(col).cast(pl.String).is_in(list(rare_set)))
                .then(pl.lit(RARE_TOKEN))
                .otherwise(pl.col(col).cast(pl.String))
                .alias(col)
            )
        return df_clean

    def explain(self) -> str:
        return (
            f"RareCategoryPlugin groups categories below {self._min_frequency:.1%} frequency "
            f"into {RARE_TOKEN}, capping one-hot width. Configure via "
            "PipelineConfig(rare_category_min_frequency=...)."
        )
