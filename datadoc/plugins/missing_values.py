from typing import Any
import polars as pl
from datadoc.plugins.base import BasePlugin


class MissingValuePlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "MissingValuePlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects and imputes missing values using Median (numeric) and Mode (categorical)."

    @property
    def priority(self) -> int:
        return 10  # Should run first

    def analyze(self, df: pl.DataFrame) -> dict:
        cols_with_missing = {c: df[c].null_count() for c in df.columns if df[c].null_count() > 0}
        total_missing = sum(cols_with_missing.values())
        return {
            "has_missing_values": bool(total_missing > 0),
            "total_missing": int(total_missing),
            "columns_affected": cols_with_missing,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_missing_values"):
            total = analysis_result["total_missing"]
            cols = analysis_result.get("columns_affected", {})
            col_detail = ", ".join([f"{c} ({v})" for c, v in cols.items()])
            recs.append(
                f"Found {total} missing values in: {col_detail}. "
                f"Recommendation: Impute numeric with Median and categorical with Mode."
            )
        return recs

    def fit(self, df: pl.DataFrame) -> dict:
        imputations: dict[str, Any] = {}
        for col in df.columns:
            series = df[col]
            if series.null_count() > 0:
                if series.dtype.is_numeric():
                    med = series.median()
                    if med is not None:
                        imputations[col] = float(med)
                else:
                    modes = series.drop_nulls().mode()
                    if len(modes) > 0:
                        imputations[col] = modes[0]
        return {"imputations": imputations}

    def transform(self, df: pl.DataFrame, state: dict | None = None) -> pl.DataFrame:
        state = state or {}
        imputations = state.get("imputations")
        if imputations is None:
            # Fallback to computing on the fly if transform called without prior fit
            imputations = self.fit(df).get("imputations", {})

        exprs = [
            pl.col(col).fill_null(val)
            for col, val in imputations.items()
            if col in df.columns
        ]
        return df.with_columns(exprs) if exprs else df

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.transform(df, state=None)

    def explain(self) -> str:
        return (
            "MissingValuePlugin fills missing numeric values with the column median "
            "and missing categorical values with the column mode (most frequent value)."
        )
