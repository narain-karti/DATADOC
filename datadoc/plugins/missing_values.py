import pandas as pd
from datadoc.plugins.base import BasePlugin

class MissingValuePlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "MissingValuePlugin"
        
    def analyze(self, df: pd.DataFrame) -> dict:
        missing_count = df.isnull().sum().sum()
        return {
            "has_missing_values": missing_count > 0,
            "total_missing": int(missing_count)
        }
        
    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_missing_values"):
            recs.append(f"Found {analysis_result['total_missing']} missing values. Recommendation: Impute numeric with Median and categorical with Mode.")
        return recs
        
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        # Very simple MVP strategy: Fill numeric with median, object with mode
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
        return df_clean
