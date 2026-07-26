import pandas as pd
import numpy as np
from datadoc.plugins.base import BasePlugin

class OutlierPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "OutlierPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects outliers using IQR and clips extreme values at 5th/95th percentiles."

    @property
    def priority(self) -> int:
        return 20  # After missing values

    @property
    def supported_datatypes(self) -> list:
        return ["numeric"]

    @property
    def dependencies(self) -> list:
        return ["MissingValuePlugin"]

    def analyze(self, df: pd.DataFrame) -> dict:
        outlier_info = {}
        num_cols = df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            if count > 0:
                outlier_info[col] = count

        return {
            "has_outliers": len(outlier_info) > 0,
            "outlier_columns": list(outlier_info.keys()),
            "outlier_counts": outlier_info,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_outliers"):
            info = analysis_result.get("outlier_counts", {})
            detail = ", ".join([f"{c} ({v} outliers)" for c, v in info.items()])
            recs.append(
                f"Outliers detected in: {detail}. "
                f"Recommendation: Clip values at 5th and 95th percentiles."
            )
        return recs

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_outliers"):
            return ""
        cols_str = str(analysis_result.get("outlier_columns", []))
        return f"""# Outlier Clipping
outlier_cols = {cols_str}
for col in outlier_cols:
    lower = df[col].quantile(0.05)
    upper = df[col].quantile(0.95)
    df[col] = df[col].clip(lower=lower, upper=upper)"""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        outlier_cols = self.analyze(df_clean).get("outlier_columns", [])
        for col in outlier_cols:
            lower = df_clean[col].quantile(0.05)
            upper = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        return df_clean

    def validate(self, df: pd.DataFrame) -> bool:
        # After clipping, no values should be outside 5th/95th range
        return not df.select_dtypes(include=[np.number]).empty

    def explain(self) -> str:
        return (
            "OutlierPlugin detects statistical outliers using the IQR (Interquartile Range) "
            "method and clips extreme values to the 5th and 95th percentile boundaries."
        )
