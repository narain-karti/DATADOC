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

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_missing_values"):
            return ""
        return """# Missing Value Imputation
for col in df.columns:
    if df[col].null_count() > 0:
        if df[col].dtype.is_numeric():
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).median()))
        else:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).drop_nulls().mode().first()))"""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        for col in df_clean.columns:
            if df_clean[col].null_count() > 0:
                if df_clean[col].dtype.is_numeric():
                    df_clean = df_clean.with_columns(pl.col(col).fill_null(pl.col(col).median()))
                else:
                    df_clean = df_clean.with_columns(pl.col(col).fill_null(pl.col(col).drop_nulls().mode().first()))
        return df_clean



    def explain(self) -> str:
        return (
            "MissingValuePlugin fills missing numeric values with the column median "
            "and missing categorical values with the column mode (most frequent value)."
        )
