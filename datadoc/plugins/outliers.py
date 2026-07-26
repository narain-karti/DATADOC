import pandas as pd
import numpy as np
from datadoc.plugins.base import BasePlugin

class OutlierPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "OutlierPlugin"
        
    def analyze(self, df: pd.DataFrame) -> dict:
        outlier_cols = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            # Simple IQR detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Check if any values are outside the fences
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_cols.append(col)
                
        return {
            "has_outliers": len(outlier_cols) > 0,
            "outlier_columns": outlier_cols
        }
        
    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_outliers"):
            cols = analysis_result["outlier_columns"]
            recs.append(f"Found outliers in {len(cols)} columns ({', '.join(cols)}). Recommendation: Clip values at 5th and 95th percentiles.")
        return recs

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        outlier_cols = self.analyze(df_clean).get("outlier_columns", [])
        for col in outlier_cols:
            lower = df_clean[col].quantile(0.05)
            upper = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        return df_clean
