import polars as pl
from datadoc.plugins.base import BasePlugin


class OutlierPlugin(BasePlugin):
    def __init__(self, outlier_multiplier: float = 1.5):
        self._outlier_multiplier = outlier_multiplier
        super().__init__()

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
            lower = Q1 - self._outlier_multiplier * IQR
            upper = Q3 + self._outlier_multiplier * IQR

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

    def fit(self, df: pl.DataFrame) -> dict:
        bounds = self._get_iqr_bounds(df)
        clip_bounds = {col: {"lower": float(lower), "upper": float(upper)} for col, (lower, upper, _count) in bounds.items()}
        return {"clip_bounds": clip_bounds}

    def transform(self, df: pl.DataFrame, state: dict | None = None) -> pl.DataFrame:
        state = state or {}
        clip_bounds = state.get("clip_bounds")
        if clip_bounds is None:
            clip_bounds = self.fit(df).get("clip_bounds", {})

        exprs = [
            pl.col(col).clip(bounds["lower"], bounds["upper"])
            for col, bounds in clip_bounds.items()
            if col in df.columns
        ]
        return df.with_columns(exprs) if exprs else df

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.transform(df, state=None)

    def explain(self) -> str:
        return (
            f"OutlierPlugin detects statistical outliers using the IQR (Interquartile Range) "
            f"method and caps extreme values at IQR boundaries (Q1 - {self._outlier_multiplier}*IQR, Q3 + {self._outlier_multiplier}*IQR)."
        )
