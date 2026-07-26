import pandas as pd
import numpy as np
from datadoc.plugins.base import BasePlugin

class ScalingPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "ScalingPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects numeric columns with vastly different scales and applies StandardScaler normalization."

    @property
    def priority(self) -> int:
        return 45  # Near the end, after encoding

    @property
    def supported_datatypes(self) -> list:
        return ["numeric"]

    @property
    def dependencies(self) -> list:
        return ["MissingValuePlugin", "OutlierPlugin"]

    def analyze(self, df: pd.DataFrame) -> dict:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return {"has_scale_issues": False, "columns_to_scale": []}

        stds = df[num_cols].std()
        # If the ratio between max std and min std is > 10, scaling is needed
        if stds.min() == 0:
            cols_to_scale = [c for c in num_cols if stds[c] > 0]
        else:
            ratio = stds.max() / stds.min()
            cols_to_scale = num_cols if ratio > 10 else []

        return {
            "has_scale_issues": len(cols_to_scale) > 0,
            "columns_to_scale": cols_to_scale,
            "scale_ratio": round(float(stds.max() / stds.min()), 2) if stds.min() > 0 else float('inf'),
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_scale_issues"):
            ratio = analysis_result.get("scale_ratio", 0)
            cols = analysis_result["columns_to_scale"]
            recs.append(
                f"Numeric columns have a scale ratio of {ratio}x across {len(cols)} columns. "
                f"Recommendation: Apply Standard Scaling (zero mean, unit variance)."
            )
        return recs

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_scale_issues"):
            return ""
        cols_str = str(analysis_result.get("columns_to_scale", []))
        return f"""# Standard Scaling
scale_cols = {cols_str}
for col in scale_cols:
    mean = df[col].mean()
    std = df[col].std()
    if std > 0:
        df[col] = (df[col] - mean) / std"""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        cols_to_scale = self.analyze(df_clean).get("columns_to_scale", [])
        for col in cols_to_scale:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                df_clean[col] = (df_clean[col] - mean) / std
        return df_clean

    def validate(self, df: pd.DataFrame) -> bool:
        return True

    def explain(self) -> str:
        return (
            "ScalingPlugin detects when numeric columns have vastly different scales "
            "(std ratio > 10x) and applies Standard Scaling: (value - mean) / std, "
            "resulting in zero-mean, unit-variance features."
        )
