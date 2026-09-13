from __future__ import annotations

from typing import Any
import polars as pl
from datadoc.plugins.base import BasePlugin


class TargetEncoderPlugin(BasePlugin):
    """Calculates smoothed target encoding for categorical columns using empirical Bayes m-estimate.

    Formula:
        y_hat = (n * mean_target + m * global_mean) / (n + m)

    Where:
        n = count of category occurrences
        mean_target = average target value for category
        m = smoothing factor (prior weight, default 10.0)
        global_mean = overall dataset target mean

    Priority 41 places it after CategoricalEncoderPlugin (40) and before RareCategoryPlugin (42).
    """

    def __init__(
        self,
        target: str | None = None,
        smoothing: float = 10.0,
        min_cardinality: int = 2,
        max_cardinality: int = 1000,
    ) -> None:
        self._target = target
        self._smoothing = smoothing
        self._min_cardinality = min_cardinality
        self._max_cardinality = max_cardinality
        super().__init__()

    @property
    def name(self) -> str:
        return "TargetEncoderPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return (
            "Calculates smoothed target encoding for categorical columns using empirical "
            "Bayes m-estimate smoothing to prevent target leakage and one-hot explosion."
        )

    @property
    def priority(self) -> int:
        return 41

    def _find_target(self, df: pl.DataFrame) -> str | None:
        if self._target and self._target in df.columns:
            return self._target
        for candidate in ("target", "label", "survived", "churn", "y", "outcome"):
            for col in df.columns:
                if col.lower() == candidate and (
                    df[col].dtype.is_numeric() or df[col].dtype == pl.Boolean
                ):
                    return col
        return None

    def analyze(self, df: pl.DataFrame) -> dict[str, Any]:
        target = self._find_target(df)
        if not target:
            return {
                "has_target": False,
                "target": None,
                "candidate_columns": [],
                "cardinalities": {},
                "smoothing": self._smoothing,
            }

        candidates = []
        cardinalities = {}
        for col in df.columns:
            if col == target:
                continue
            if df[col].dtype == pl.String:
                n_uniq = df[col].drop_nulls().n_unique()
                if self._min_cardinality <= n_uniq <= self._max_cardinality:
                    candidates.append(col)
                    cardinalities[col] = n_uniq

        return {
            "has_target": True,
            "target": target,
            "candidate_columns": candidates,
            "cardinalities": cardinalities,
            "smoothing": self._smoothing,
        }

    def recommend(self, analysis_result: dict[str, Any]) -> list[str]:
        if not analysis_result.get("has_target") or not analysis_result.get("candidate_columns"):
            return []
        candidates = analysis_result["candidate_columns"]
        target = analysis_result["target"]
        m = analysis_result["smoothing"]
        return [
            f"Target encoding candidate columns detected: {', '.join(candidates)} with target '{target}'. "
            f"Recommendation: Apply empirical Bayes target encoding (smoothing m={m}) to capture non-linear signal "
            f"without dimensionality explosion."
        ]

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        analysis = self.analyze(df)
        if not analysis.get("has_target") or not analysis.get("candidate_columns"):
            return df

        target = analysis["target"]
        candidates = analysis["candidate_columns"]
        m = self._smoothing

        df_out = df.clone()
        target_series = df_out[target]
        if target_series.dtype == pl.Boolean:
            target_series = target_series.cast(pl.Float64)
        elif not target_series.dtype.is_numeric():
            return df_out

        global_mean = float(target_series.drop_nulls().mean() or 0.0)

        for col in candidates:
            stats = (
                df_out.select([col, target])
                .filter(pl.col(target).is_not_null())
                .group_by(col)
                .agg(
                    [
                        pl.len().alias("_dd_count"),
                        pl.col(target).cast(pl.Float64).mean().alias("_dd_cat_mean"),
                    ]
                )
            )

            stats = stats.with_columns(
                (
                    (pl.col("_dd_count") * pl.col("_dd_cat_mean") + pl.lit(m * global_mean))
                    / (pl.col("_dd_count") + pl.lit(m))
                ).alias(f"{col}__target_enc")
            ).select([col, f"{col}__target_enc"])

            df_out = df_out.join(stats, on=col, how="left")
            df_out = df_out.with_columns(pl.col(f"{col}__target_enc").fill_null(global_mean))

        return df_out

    def explain(self) -> str:
        return (
            f"TargetEncoderPlugin computes smoothed target encoding for categorical features: "
            f"y_hat = (n * cat_mean + {self._smoothing} * global_mean) / (n + {self._smoothing}). "
            f"Prior weighting prevents overfitting on rare categories and handles unseen test categories."
        )
