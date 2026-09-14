"""Prompt engineering templates and statistical profile digest generator for DATADOC AI."""

from __future__ import annotations

from typing import Any, Optional
import polars as pl


def build_profile_digest(
    profile_dict: dict[str, Any],
    df: pl.DataFrame,
    target: Optional[str] = None,
) -> str:
    """Builds an anonymized, token-efficient summary digest of the dataset statistics.

    Contains column names, inferred data types, cardinality, missing percentages,
    numeric 5-point distributions, top categorical frequencies, and sample values.
    Keeps token count compact (~1,000 - 2,000 tokens) regardless of dataset row count.
    """
    rows = df.height
    cols = df.width
    null_counts = profile_dict.get("null_counts", {})
    unique_counts = profile_dict.get("cardinality", {})
    col_types = profile_dict.get("schema", {})
    findings = profile_dict.get("findings", [])

    lines: list[str] = []
    lines.append(f"DATASET SUMMARY: {rows:,} rows, {cols} columns.")
    if target:
        lines.append(f"PREDICTIVE TARGET COLUMN: '{target}'")
    else:
        lines.append("PREDICTIVE TARGET: Not explicitly specified.")

    lines.append("\nCOLUMN DETAILS:")
    for col in df.columns:
        c_type = col_types.get(col, str(df[col].dtype))
        n_null = null_counts.get(col, 0)
        pct_null = (n_null / rows * 100) if rows else 0
        n_unique = unique_counts.get(col, df[col].n_unique())

        col_desc = f"- Column '{col}': Type={c_type}, Unique={n_unique:,}, Missing={n_null:,} ({pct_null:.1f}%)"

        # If numeric, add summary stats
        if (
            c_type in {"numeric", "float", "integer", "Int64", "Float64"}
            or df[col].dtype.is_numeric()
        ):
            non_nulls = df[col].drop_nulls()
            if len(non_nulls) > 0:
                c_min = float(non_nulls.min()) if non_nulls.min() is not None else 0  # type: ignore[arg-type]
                c_max = float(non_nulls.max()) if non_nulls.max() is not None else 0  # type: ignore[arg-type]
                c_med = float(non_nulls.median()) if non_nulls.median() is not None else 0  # type: ignore[arg-type]
                c_mean = float(non_nulls.mean()) if non_nulls.mean() is not None else 0  # type: ignore[arg-type]
                col_desc += (
                    f", Range=[{c_min:.2f} .. {c_max:.2f}], Median={c_med:.2f}, Mean={c_mean:.2f}"
                )

        # If categorical or string, show top frequent samples
        elif (
            c_type in {"categorical", "string", "String", "Utf8"}
            or df[col].dtype in (pl.String, pl.Categorical)
            or "String" in str(df[col].dtype)
            or "Utf8" in str(df[col].dtype)
        ):
            top_vals = df[col].drop_nulls().value_counts().sort(by="count", descending=True).head(3)
            samples = [f"'{row[0]}' ({row[1]})" for row in top_vals.iter_rows()]
            if samples:
                col_desc += f", TopValues=[{', '.join(samples)}]"

        lines.append(col_desc)

    if findings:
        lines.append("\nDETECTED DATA QUALITY FINDINGS:")
        for f in findings[:15]:
            lines.append(
                f"- [{f.get('severity', 'info').upper()}] {f.get('code')}: {f.get('message')}"
            )

    return "\n".join(lines)


SYSTEM_PROMPT = """You are DATADOC's Principal Applied ML & Feature Engineering Expert.
Your purpose is to deeply analyze tabular dataset profiles to help data scientists and ML engineers build the highest-performing, leakage-safe machine learning models.

CRITICAL INSTRUCTIONS:
1. Provide actionable, rigorous, and domain-grounded advice.
2. Formulate concrete mathematical feature engineering formulas (e.g. ratios, aggregations, non-linear interaction terms).
3. Identify hidden missing-value sentinels (e.g. 999, -1, 9999, '?', 'unknown').
4. Diagnose missingness mechanisms (MCAR vs MAR vs MNAR) and warn against target leakage vectors.
5. Format your output strictly in the four markdown sections requested. Be concise, punchy, and highly analytical."""


def build_analysis_prompt(digest: str, target: Optional[str] = None) -> str:
    """Constructs the user analysis prompt with the statistical digest."""
    target_clause = (
        f"The target variable for machine learning is '{target}'."
        if target
        else "No specific target variable was designated."
    )

    return f"""Please provide a comprehensive feature engineering and data quality advisory for this dataset.
{target_clause}

{digest}

Please structure your response into EXACTLY these 4 sections:

### 1. Semantic Column Profiling & Hidden Sentinels
- Analyze column names and sample values to deduce their true real-world meaning and business role.
- Flag potential hidden sentinels or placeholder values (e.g., extreme integers like -999, 999, or strings like 'None', 'missing', '?') masquerading as valid data.
- Distinguish true identifiers/tracking codes from ordinal or continuous features.

### 2. Missingness Mechanisms & Target Leakage Risks
- Evaluate missingness mechanisms: Is missing data Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR)?
- Highlight if missingness in any column carries predictive signal where a binary indicator flag (`col_is_missing`) is essential.
- Scrutinize columns for potential Target Leakage (features that represent post-outcome actions, high-cardinality spurious IDs, or temporal future leakage).

### 3. Domain Feature Engineering Hypotheses
- Propose 3 to 5 specific, high-impact feature transformations (ratios, interaction products, temporal spans, log-transforms, grouping bins) tailored to this specific domain.
- For each feature, write the exact mathematical or Polars expression (e.g., `pl.col('A') / (pl.col('B') + 1)`).
- Explain WHY this feature provides predictive lift for tree-based models (XGBoost/LightGBM) or linear models.

### 4. Recommended DATADOC Transformation Strategy
- Preset Recommendation: Recommend whether `tree`, `linear`, or `balanced` is optimal.
- Categorical Strategy: Recommend One-Hot Encoding vs. Empirical Bayes Target Encoding vs. Rare Category grouping.
- Outlier & Scaling Guidance: Specific IQR bounds or standard scaling advice.
"""
