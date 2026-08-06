import polars as pl
from datadoc.plugins.base import BasePlugin

class CategoricalEncoderPlugin(BasePlugin):
    def __init__(self, max_categories: int = 10):
        self._max_categories = max_categories
        super().__init__()

    @property
    def name(self) -> str:
        return "CategoricalEncoderPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Drops high-cardinality identifier columns and applies One-Hot Encoding (drop_first=True) to low-cardinality categoricals."

    @property
    def priority(self) -> int:
        return 40  # After missing values and outliers

    def analyze(self, df: pl.DataFrame) -> dict:
        str_cols = [col for col in df.columns if df[col].dtype == pl.String]

        # Identifier columns: every value is unique (e.g. Name, Email)
        id_cols = [col for col in str_cols if df[col].drop_nulls().n_unique() >= df.height]

        # Encodable categorical: low cardinality (2-max_categories unique), not an identifier
        valid_cats = [
            col for col in str_cols
            if col not in id_cols and 1 < df[col].n_unique() < self._max_categories
        ]
        cardinality = {col: df[col].n_unique() for col in valid_cats}

        return {
            "has_categorical": len(valid_cats) > 0 or len(id_cols) > 0,
            "categorical_columns": valid_cats,
            "identifier_columns": id_cols,
            "cardinality": cardinality,
        }

    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        id_cols = analysis_result.get("identifier_columns", [])
        if id_cols:
            recs.append(
                f"Identifier columns detected: {', '.join(id_cols)}. "
                f"Recommendation: Drop these (every value is unique, no ML signal)."
            )
        if analysis_result.get("categorical_columns"):
            info = analysis_result.get("cardinality", {})
            detail = ", ".join([f"{c} ({v} unique)" for c, v in info.items()])
            recs.append(
                f"Categorical columns found: {detail}. "
                f"Recommendation: Apply One-Hot Encoding."
            )
        return recs

    def generate_code(self, analysis_result: dict) -> str:
        lines = []
        id_cols = analysis_result.get("identifier_columns", [])
        if id_cols:
            lines.append(f"# Drop identifier columns")
            lines.append(f"df = df.drop({id_cols})")

        cat_cols = analysis_result.get("categorical_columns", [])
        if cat_cols:
            cols_str = str(cat_cols)
            lines.append(f"# Categorical Encoding (One-Hot)")
            lines.append(f"cat_cols = {cols_str}")
            lines.append(f"df = df.to_dummies(columns=cat_cols, drop_first=True)")

        return "\n".join(lines)

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        analysis = self.analyze(df_clean)

        # Drop identifier columns first
        id_cols = analysis.get("identifier_columns", [])
        if id_cols:
            df_clean = df_clean.drop(id_cols)

        # One-hot encode low-cardinality categoricals
        cat_cols = analysis.get("categorical_columns", [])
        if cat_cols:
            df_clean = df_clean.to_dummies(columns=cat_cols, drop_first=True)
        return df_clean

    def explain(self) -> str:
        return (
            f"CategoricalEncoderPlugin detects text/string columns. "
            f"High-cardinality identifier columns (every value unique) are dropped. "
            f"Low-cardinality columns (< {self._max_categories} unique values) are One-Hot Encoded with drop_first=True."
        )

