import pandas as pd
from datadoc.plugins.base import BasePlugin

class CategoricalEncoderPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "CategoricalEncoderPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Detects categorical columns and applies One-Hot Encoding (drop_first=True)."

    @property
    def priority(self) -> int:
        return 40  # After missing values and outliers

    @property
    def supported_datatypes(self) -> list:
        return ["categorical"]

    @property
    def dependencies(self) -> list:
        return ["MissingValuePlugin"]

    def analyze(self, df: pd.DataFrame) -> dict:
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        valid_cats = [col for col in cat_cols if df[col].nunique() < 10 and df[col].nunique() > 1]
        cardinality = {col: int(df[col].nunique()) for col in valid_cats}

        return {
            "has_categorical": len(valid_cats) > 0,
            "categorical_columns": valid_cats,
            "cardinality": cardinality,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_categorical"):
            info = analysis_result.get("cardinality", {})
            detail = ", ".join([f"{c} ({v} unique)" for c, v in info.items()])
            recs.append(
                f"Categorical columns found: {detail}. "
                f"Recommendation: Apply One-Hot Encoding."
            )
        return recs

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_categorical"):
            return ""
        cols_str = str(analysis_result.get("categorical_columns", []))
        return f"""# Categorical Encoding (One-Hot)
cat_cols = {cols_str}
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)"""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        cat_cols = self.analyze(df_clean).get("categorical_columns", [])
        if cat_cols:
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
        return df_clean

    def validate(self, df: pd.DataFrame) -> bool:
        # After encoding, no object columns should remain (for the encoded ones)
        return True

    def explain(self) -> str:
        return (
            "CategoricalEncoderPlugin detects text/object columns with fewer than 10 unique values "
            "and applies One-Hot Encoding with drop_first=True to avoid multicollinearity."
        )
