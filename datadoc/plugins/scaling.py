import polars as pl
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



    def analyze(self, df: pl.DataFrame) -> dict:
        num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
        if len(num_cols) < 2:
            return {"has_scale_issues": False, "columns_to_scale": []}

        stds = {}
        for col in num_cols:
            std = df[col].std()
            if std is not None:
                stds[col] = std
                
        if not stds:
            return {"has_scale_issues": False, "columns_to_scale": []}

        min_std = min(stds.values())
        max_std = max(stds.values())

        if min_std == 0:
            cols_to_scale = [c for c, s in stds.items() if s > 0]
            ratio = float('inf')
        else:
            ratio = max_std / min_std
            cols_to_scale = num_cols if ratio > 10 else []

        return {
            "has_scale_issues": len(cols_to_scale) > 0,
            "columns_to_scale": cols_to_scale,
            "scale_ratio": round(ratio, 2) if ratio != float('inf') else ratio,
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
exprs = [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in scale_cols]
if exprs:
    df = df.with_columns(exprs)"""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        cols_to_scale = self.analyze(df_clean).get("columns_to_scale", [])
        
        exprs = []
        for col in cols_to_scale:
            exprs.append(
                ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
            )
            
        if exprs:
            df_clean = df_clean.with_columns(exprs)
            
        return df_clean



    def explain(self) -> str:
        return (
            "ScalingPlugin detects when numeric columns have vastly different scales "
            "(std ratio > 10x) and applies Standard Scaling: (value - mean) / std, "
            "resulting in zero-mean, unit-variance features."
        )
