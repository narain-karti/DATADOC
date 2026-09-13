from __future__ import annotations

import itertools
from typing import Any
import polars as pl
from datadoc.plugins.base import BasePlugin


class PolynomialFeaturesPlugin(BasePlugin):
    """Generates degree-2 interaction terms (x1 * x2) and squared terms (x^2) for numerical features.

    Captures non-linear relationships and feature interactions for linear models
    (LogisticRegression, Ridge, Lasso) without manual feature engineering.

    Priority 44 places it after categorical/rare encoders and right before ScalingPlugin (45).
    """

    def __init__(
        self,
        max_numeric_features: int = 5,
        include_squared: bool = True,
        include_interactions: bool = True,
        target: str | None = None,
    ) -> None:
        self._max_numeric_features = max_numeric_features
        self._include_squared = include_squared
        self._include_interactions = include_interactions
        self._target = target
        super().__init__()

    @property
    def name(self) -> str:
        return "PolynomialFeaturesPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return (
            "Generates degree-2 interaction terms (x1 * x2) and squared terms (x^2) "
            "for numerical features to model non-linear effects."
        )

    @property
    def priority(self) -> int:
        return 44

    def _get_candidates(self, df: pl.DataFrame) -> list[str]:
        candidates = []
        for col in df.columns:
            if self._target and col == self._target:
                continue
            # Skip binary / indicator columns (e.g. *__missing)
            if col.endswith("__missing") or col.endswith("__sin") or col.endswith("__cos"):
                continue
            s = df[col]
            if s.dtype.is_numeric() and s.drop_nulls().n_unique() > 2:
                candidates.append(col)
        # Cap at max_numeric_features to avoid feature explosion
        return candidates[: self._max_numeric_features]

    def analyze(self, df: pl.DataFrame) -> dict[str, Any]:
        candidates = self._get_candidates(df)
        pairs = list(itertools.combinations(candidates, 2))
        return {
            "has_numeric": len(candidates) >= 2,
            "candidate_columns": candidates,
            "pairs_count": len(pairs) if self._include_interactions else 0,
            "squared_count": len(candidates) if self._include_squared else 0,
        }

    def recommend(self, analysis_result: dict[str, Any]) -> list[str]:
        if not analysis_result.get("has_numeric"):
            return []
        cols = analysis_result["candidate_columns"]
        n_pairs = analysis_result["pairs_count"]
        n_sq = analysis_result["squared_count"]
        return [
            f"Numerical features detected for interaction expansion: {', '.join(cols)}. "
            f"Recommendation: Generate {n_pairs} pairwise interaction terms (x1 * x2) and {n_sq} squared terms (x^2) "
            f"to capture non-linear effects before scaling."
        ]

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        candidates = self._get_candidates(df)
        if len(candidates) < 2:
            return df

        new_exprs = []
        # Squared terms: x^2
        if self._include_squared:
            for col in candidates:
                new_exprs.append((pl.col(col) * pl.col(col)).alias(f"{col}__pow2"))

        # Pairwise interactions: x1 * x2
        if self._include_interactions:
            for c1, c2 in itertools.combinations(candidates, 2):
                new_exprs.append((pl.col(c1) * pl.col(c2)).alias(f"{c1}__x__{c2}"))

        if new_exprs:
            return df.with_columns(new_exprs)
        return df

    def explain(self) -> str:
        return (
            f"PolynomialFeaturesPlugin generates degree-2 terms for up to {self._max_numeric_features} "
            f"numerical features: pairwise products (x1 * x2) and squares (x^2). "
            f"This captures multiplicative and quadratic signals, boosting linear estimator expressiveness."
        )
