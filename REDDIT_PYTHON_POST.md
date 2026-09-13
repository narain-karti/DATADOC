<!-- 
===================================================================
REDDIT R/PYTHON SUBMISSION FILE
===================================================================

FLAIR: Select "Showcase"
TYPE: Text Post

TITLE TO COPY (72 characters):
DATADOC: Fast, zero-leakage tabular ML dataset preparation built on Polars

Everything below the horizontal divider is the POST BODY to copy & paste.
===================================================================
-->

### What My Project Does

**DATADOC** is an open-source CLI and Python library that automates tabular dataset engineering and feature preparation for machine learning, while strictly enforcing zero data leakage.

It is built entirely on the **Polars** expression engine and Apache Arrow memory layout.

- **GitHub**: https://github.com/narain-karti/DATADOC
- **PyPI**: `pip install datadoc-cli`
- **Documentation**: https://narain-karti.github.io/DATADOC/

Instead of maintaining ad-hoc pandas scripts in Jupyter notebooks, DATADOC separates data preparation into a strict, reproducible 4-stage lifecycle:
1. `profile`: Computes null percentages, schema fingerprints, and infers column roles (target, continuous, categorical, datetime, identifier).
2. `plan`: Generates a deterministic transformation plan before mutating any data.
3. `fit`: Learns imputation medians, categorical vocabularies, Tukey IQR outlier limits, and scaling statistics **strictly on the training split**.
4. `transform`: Validates input contracts and applies the frozen state to validation, test, or inference data.
5. `save / load`: Serializes the complete state into an inspectable JSON artifact (`pipeline.json`) instead of an opaque Python pickle file.

#### Python SDK Example:
```python
import polars as pl
from datadoc import DataDocPipeline, PipelineConfig

train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

config = PipelineConfig(
    target="churn",
    drop_identifiers=True,
    deduplicate=True,
    clip_outliers=True,
    scaling="standard",
    rare_category_min_frequency=0.01,
)

# Fit strictly on train split:
pipeline = DataDocPipeline(config).fit(train_df)

# Transform unseen test data without distribution bleed:
clean_train = pipeline.transform(train_df)
clean_test = pipeline.transform(test_df)

# Serialize state for production API serving:
pipeline.save("artifacts/pipeline.json")
```

#### CLI Usage:
```bash
# 1. Instant data health audit (0-100 score & quality grade):
datadoc health train.csv --target churn

# 2. Fit train-only state and export pipeline:
datadoc fit train.csv --target churn --preset balanced --output artifacts/pipeline.json

# 3. Transform new data with drift validation:
datadoc transform test.csv --pipeline artifacts/pipeline.json --output clean_test.csv --validate

# 4. Generate standalone interactive HTML audit report:
datadoc report train.csv --target churn --output report.html
```

---

### Target Audience

DATADOC is aimed at:
- **Data Scientists & ML Engineers** who spend hours copy-pasting pandas preprocessing code across notebooks and want a fast, reproducible pipeline.
- **Production Teams** needing to hand off feature engineering from exploration to production microservices without train/serve skew or pickle dependency vulnerabilities.
- **Kaggle / Competitive ML Practitioners** who want to eliminate accidental train/test leakage that degrades private leaderboard scores.

---

### Comparison

| Feature | DATADOC | Pandas Ad-Hoc Scripts | scikit-learn `Pipeline` |
| :--- | :--- | :--- | :--- |
| **Engine** | Vectorized Polars (Apache Arrow) | Pandas (copy-on-write overhead) | NumPy conversions |
| **Artifact Format** | Inspectable `pipeline.json` | None (code re-execution) | Python `pickle` (version-fragile, security risks) |
| **Leakage Protection** | Strict train-only `.fit()` boundary | Easy to accidentally leak medians/vocab | Manual slicing required |
| **Outlier Handling** | Train-fitted Tukey IQR clipping | Manual code | Custom transformers |
| **Schema Validation** | Validates contract & alerts on drift | Silent coercion or crashes | Relies on array dimensions |
| **Reporting** | Standalone interactive HTML reports | External packages required | None built-in |

---

The project is MIT licensed, has 110 automated tests, and runs locally with zero telemetry.

I would love feedback from the community on the plugin registry architecture and artifact serialization format!
