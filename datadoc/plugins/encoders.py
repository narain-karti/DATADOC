import polars as pl
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

    def analyze(self, df: pl.DataFrame) -> dict:
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        valid_cats = [col for col in cat_cols if df[col].n_unique() < 10 and df[col].n_unique() > 1]
        cardinality = {col: df[col].n_unique() for col in valid_cats}

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
df = df.to_dummies(columns=cat_cols, drop_first=True)"""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        cat_cols = self.analyze(df_clean).get("categorical_columns", [])
        if cat_cols:
            df_clean = df_clean.to_dummies(columns=cat_cols, drop_first=True)
        return df_clean

    def validate(self, df: pl.DataFrame) -> bool:
        return True

    def explain(self) -> str:
        return (
            "CategoricalEncoderPlugin detects text/string columns with fewer than 10 unique values "
            "and applies One-Hot Encoding with drop_first=True to avoid multicollinearity."
        )
