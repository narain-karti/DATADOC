import pandas as pd
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

    @property
    def supported_datatypes(self) -> list:
        return ["datetime"]

    @property
    def dependencies(self) -> list:
        return ["MissingValuePlugin"]

    def analyze(self, df: pd.DataFrame) -> dict:
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(df[col], errors='coerce')
                        # If more than 50% parsed successfully, treat as datetime
                        if parsed.notna().mean() > 0.5:
                            datetime_cols.append(col)
                except Exception:
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
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day
    df[col + '_dayofweek'] = df[col].dt.dayofweek
    df = df.drop(columns=[col])"""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        dt_cols = self.analyze(df_clean).get("datetime_columns", [])

        for col in dt_cols:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            df_clean[col + '_year'] = df_clean[col].dt.year
            df_clean[col + '_month'] = df_clean[col].dt.month
            df_clean[col + '_day'] = df_clean[col].dt.day
            df_clean[col + '_dayofweek'] = df_clean[col].dt.dayofweek
            df_clean = df_clean.drop(columns=[col])

        return df_clean

    def validate(self, df: pd.DataFrame) -> bool:
        return True

    def explain(self) -> str:
        return (
            "DatetimePlugin detects columns containing dates/times and extracts "
            "year, month, day, and day_of_week as new numeric features. "
            "The original datetime column is dropped."
        )
