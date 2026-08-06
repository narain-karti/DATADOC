import polars as pl
from datadoc.plugins.base import BasePlugin


class ScalingPlugin(BasePlugin):
    def __init__(self, scaling_ratio: float = 10.0):
        self._scaling_ratio = scaling_ratio
        super().__init__()

    @property
    def name(self) -> str:
        return "ScalingPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects continuous numeric columns with vastly different scales and applies StandardScaler normalization."

    @property
    def priority(self) -> int:
        return 45  # Near the end, after encoding

    @staticmethod
    def _is_binary(series: pl.Series) -> bool:
        """Check if a column contains only two distinct values (e.g. 0/1 from one-hot encoding)."""
        unique = series.drop_nulls().unique()
        return unique.len() <= 2

    @staticmethod
    def _get_scalable_columns(df: pl.DataFrame) -> list[str]:
        """Return numeric columns that are continuous (not binary, not constant)."""
        scalable = []
        for col in df.columns:
            if not df[col].dtype.is_numeric():
                continue
            series = df[col].drop_nulls()
            if series.len() == 0:
                continue
            # Skip constant columns
            if series.n_unique() <= 1:
                continue
            # Skip binary columns (0/1 from one-hot encoding)
            if ScalingPlugin._is_binary(df[col]):
                continue
            scalable.append(col)
        return scalable

    def analyze(self, df: pl.DataFrame) -> dict:
        scalable_cols = self._get_scalable_columns(df)
        if len(scalable_cols) < 2:
            return {"has_scale_issues": False, "columns_to_scale": []}

        stds = {}
        for col in scalable_cols:
            std = df[col].std()
            if std is not None and std > 0:
                stds[col] = std

        if len(stds) < 2:
            return {"has_scale_issues": False, "columns_to_scale": []}

        min_std = min(stds.values())
        max_std = max(stds.values())
        ratio = max_std / min_std if min_std > 0 else float("inf")
        cols_to_scale = list(stds.keys()) if ratio > self._scaling_ratio else []

        return {
            "has_scale_issues": len(cols_to_scale) > 0,
            "columns_to_scale": cols_to_scale,
            "scale_ratio": round(ratio, 2) if ratio != float("inf") else ratio,
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
        return f"""# Standard Scaling (continuous features only)
scale_cols = {cols_str}
exprs = [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in scale_cols]
if exprs:
    df = df.with_columns(exprs)"""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        cols_to_scale = self.analyze(df_clean).get("columns_to_scale", [])

        exprs = []
        for col in cols_to_scale:
            exprs.append(((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col))

        if exprs:
            df_clean = df_clean.with_columns(exprs)

        return df_clean

    def explain(self) -> str:
        return (
            f"ScalingPlugin detects when continuous numeric columns have vastly different scales "
            f"(std ratio > {self._scaling_ratio}x) and applies Standard Scaling: (value - mean) / std, "
            f"resulting in zero-mean, unit-variance features. "
            f"Binary columns (e.g. one-hot encoded) and constant columns are excluded."
        )
