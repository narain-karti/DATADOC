import pandas as pd
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

    @property
    def supported_datatypes(self) -> list:
        return ["numeric", "categorical"]

    def analyze(self, df: pd.DataFrame) -> dict:
        missing_count = df.isnull().sum().sum()
        missing_by_col = df.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0].to_dict()
        return {
            "has_missing_values": missing_count > 0,
            "total_missing": int(missing_count),
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
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])"""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        return df_clean

    def validate(self, df: pd.DataFrame) -> bool:
        return df.isnull().sum().sum() == 0

    def explain(self) -> str:
        return (
            "MissingValuePlugin fills missing numeric values with the column median "
            "and missing categorical values with the column mode (most frequent value)."
        )
