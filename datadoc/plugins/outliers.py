import polars as pl
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
        return "Detects outliers using IQR and caps extreme values at IQR boundaries."

    @property
    def priority(self) -> int:
        return 20  # After missing values

    @property
    def dependencies(self) -> list[str]:
        return ["MissingValuePlugin"]

    def _get_iqr_bounds(self, df: pl.DataFrame) -> dict[str, tuple[float, float, int]]:
        """Calculate IQR bounds for numeric columns, returning columns with outliers."""
        bounds = {}
        num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]

        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            if Q1 is None or Q3 is None:
                continue
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            count = df.select(((pl.col(col) < lower) | (pl.col(col) > upper)).sum())[col][0]
            if count and count > 0:
                bounds[col] = (lower, upper, int(count))

        return bounds

    def analyze(self, df: pl.DataFrame) -> dict:
        bounds = self._get_iqr_bounds(df)
        outlier_info = {col: info[2] for col, info in bounds.items()}

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
                f"Recommendation: Cap values at IQR boundaries (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."
            )
        return recs

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_outliers"):
            return ""
        cols_str = str(analysis_result.get("outlier_columns", []))
        return f"""# Outlier Capping (IQR method)
outlier_cols = {cols_str}
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    if Q1 is not None and Q3 is not None:
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df.with_columns(pl.col(col).clip(lower, upper))"""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        bounds = self._get_iqr_bounds(df_clean)

        for col, (lower, upper, _count) in bounds.items():
            df_clean = df_clean.with_columns(pl.col(col).clip(lower, upper))

        return df_clean

    def explain(self) -> str:
        return (
            "OutlierPlugin detects statistical outliers using the IQR (Interquartile Range) "
            "method and caps extreme values at IQR boundaries (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."
        )

