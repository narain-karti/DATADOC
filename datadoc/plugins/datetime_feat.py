import polars as pl
from datadoc.plugins.base import BasePlugin

class DatetimePlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "DatetimePlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects datetime-like columns and extracts year, month, day, dayofweek features."

    @property
    def priority(self) -> int:
        return 30  # After missing values and outliers, before encoding



    def analyze(self, df: pl.DataFrame) -> dict:
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype in [pl.Date, pl.Datetime]:
                datetime_cols.append(col)
            elif df[col].dtype == pl.String:
                try:
                    parsed = df.select(pl.col(col).str.to_datetime(strict=False))
                    # if more than 50% are successfully parsed
                    if parsed.height > 0:
                        not_null_ratio = 1 - (parsed[col].null_count() / parsed.height)
                        if not_null_ratio > 0.5:
                            datetime_cols.append(col)
                except (ValueError, TypeError, pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
                    # Column is not a valid datetime string; skipping
                    pass

        return {
            "has_datetime": len(datetime_cols) > 0,
            "datetime_columns": datetime_cols,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_datetime"):
            cols = analysis_result["datetime_columns"]
            recs.append(
                f"Datetime columns detected: {', '.join(cols)}. "
                f"Recommendation: Extract year, month, day, day_of_week features and drop original."
            )
        return recs

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_datetime"):
            return ""
        cols_str = str(analysis_result.get("datetime_columns", []))
        return f"""# Datetime Feature Extraction
datetime_cols = {cols_str}
for col in datetime_cols:
    if df[col].dtype == pl.String:
        df = df.with_columns(pl.col(col).str.to_datetime(strict=False).alias(col))
    
    df = df.with_columns([
        pl.col(col).dt.year().alias(col + '_year'),
        pl.col(col).dt.month().alias(col + '_month'),
        pl.col(col).dt.day().alias(col + '_day'),
        pl.col(col).dt.weekday().alias(col + '_dayofweek')
    ]).drop(col)"""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        dt_cols = self.analyze(df_clean).get("datetime_columns", [])

        for col in dt_cols:
            if df_clean[col].dtype == pl.String:
                df_clean = df_clean.with_columns(pl.col(col).str.to_datetime(strict=False).alias(col))
                
            df_clean = df_clean.with_columns([
                pl.col(col).dt.year().alias(col + '_year'),
                pl.col(col).dt.month().alias(col + '_month'),
                pl.col(col).dt.day().alias(col + '_day'),
                pl.col(col).dt.weekday().alias(col + '_dayofweek')
            ]).drop(col)

        return df_clean



    def explain(self) -> str:
        return (
            "DatetimePlugin detects columns containing dates/times and extracts "
            "year, month, day, and day_of_week as new numeric features. "
            "The original datetime column is dropped."
        )
