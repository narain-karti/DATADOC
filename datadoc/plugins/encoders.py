import pandas as pd
from datadoc.plugins.base import BasePlugin

class CategoricalEncoderPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "CategoricalEncoderPlugin"
        
    def analyze(self, df: pd.DataFrame) -> dict:
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        # Filter out high cardinality or ID-like columns
        valid_cats = [col for col in cat_cols if df[col].nunique() < 10 and df[col].nunique() > 1]
        
        return {
            "has_categorical": len(valid_cats) > 0,
            "categorical_columns": valid_cats
        }
        
    def recommend(self, analysis_result: dict) -> list[str]:
        recs = []
        if analysis_result.get("has_categorical"):
            cols = analysis_result["categorical_columns"]
            recs.append(f"Found {len(cols)} categorical columns ({', '.join(cols)}). Recommendation: Apply One-Hot Encoding.")
        return recs
        
    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_categorical"):
            return ""
        cols_str = str(analysis_result.get("categorical_columns", []))
        return f"""# Categorical Encoding
cat_cols = {cols_str}
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)"""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        cat_cols = self.analyze(df_clean).get("categorical_columns", [])
        if cat_cols:
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
        return df_clean
