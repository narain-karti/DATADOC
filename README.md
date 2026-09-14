# DATADOC

**Local-first, leakage-safe tabular dataset engineering & feature preparation for machine learning.**

[![PyPI version](https://img.shields.io/pypi/v/datadoc-cli.svg?color=blue)](https://pypi.org/project/datadoc-cli/)
[![Python Versions](https://img.shields.io/pypi/pyversions/datadoc-cli.svg)](https://pypi.org/project/datadoc-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/narain-karti/DATADOC/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://narain-karti.github.io/DATADOC/)

[Documentation](https://narain-karti.github.io/DATADOC/) | [Installation](#installation) | [Quickstart](#quickstart) | [Benchmarks](#empirical-benchmarks) | [CLI Reference](#cli-commands-reference) | [Python SDK](#python-sdk) | [Architecture](#architecture--plugins)

---

## Overview

**DATADOC** is a high-performance dataset preparation engine and Python library built for applied machine learning. Powered by **Polars**, it inspects raw tabular data, audits quality anomalies, detects potential target leakage, and applies deterministic, mathematically ordered feature transformations.

Ad-hoc data cleaning scripts in Jupyter notebooks often introduce subtle data leakage, high-cardinality dimensionality explosion, and train/serve skew. DATADOC enforces strict engineering guarantees:

- **Strict Leakage Prevention**: All imputation medians, categorical vocabularies, outlier boundaries, and scaling parameters are learned exclusively from the training split (`.fit()`) and serialized into an immutable JSON artifact (`pipeline.json`).
- **High-Throughput Performance**: Vectorized execution on Apache Arrow via Polars handles millions of rows in seconds with minimal memory overhead.
- **Explainable by Design**: Generates human-readable transformation plans before applying modifications, accompanied by provenance manifests tracking every engineered column.
- **AI Feature Hypotheses (Advisory Only)**: Leverages multi-provider LLMs (OpenAI, Gemini, Anthropic, Groq, Ollama) via LiteLLM to infer column semantics, audit missingness mechanisms (MCAR vs. MAR vs. MNAR), and generate domain-specific mathematical interaction hypotheses without executing untrusted code.

---

## Empirical Benchmarks

To quantify the downstream impact of DATADOC's automated feature engineering, we evaluated raw naive preprocessing against DATADOC preparation across 5-fold Stratified Cross-Validation on standard benchmarks:

### Titanic Survival (Binary Classification)

| Model | Baseline (Naive Prep) | DATADOC Cleaned | Accuracy Delta | Relative Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 78.90% ± 0.99% | **79.91% ± 1.90%** | **+1.01%** | **+1.28%** |
| **Random Forest** | 82.15% ± 2.45% | **82.82% ± 2.40%** | **+0.67%** | **+0.82%** |

*Why the lift occurs*: DATADOC isolates high-cardinality string identifiers (`Name`, `Ticket`), creates informative missingness flags (`Age__missing`, `Cabin__missing`), applies Tukey-fence outlier stabilization, and enforces train-only standard scaling to eliminate distribution bleed.

---

## Installation

Install DATADOC via `pip` or `uv`:

```bash
# Core package (CLI, Polars engine, plugins, reports)
pip install datadoc-cli

# With downstream ML evaluation baselines (Scikit-Learn)
pip install "datadoc-cli[ml]"

# With local interactive web studio (FastAPI, Uvicorn)
pip install "datadoc-cli[ui]"

# With AI dataset explainer (LiteLLM)
pip install "datadoc-cli[ai]"

# Full installation (all extras)
pip install "datadoc-cli[all]"
```

*Requires Python 3.10 or higher.*

---

## Quickstart

### 1. One-Shot Automated Preparation

Execute the full pipeline—profiling, planning, fitting, transforming, and manifest generation—in a single command:

```bash
datadoc run train.csv --target churn --preset balanced --output-dir clean_train
```

### 2. Step-by-Step Auditable Workflow

For production systems requiring inspection at each stage:

```bash
# Audit data health (0-100 score, letter grade, detected quality issues)
datadoc health train.csv --target churn

# Deep statistical profile with column roles and leakage alerts
datadoc profile train.csv --target churn --explain --output profile.json

# AI-driven feature hypotheses and semantic sentinel audit (Optional AI extra)
datadoc explain train.csv --target churn --model gpt-4o-mini

# Review the explicit transformation plan before execution
datadoc plan train.csv --target churn --preset balanced --output plan.json

# Fit transformations exclusively on the training split
datadoc fit train.csv --target churn --preset balanced --output artifacts/pipeline.json

# Transform unseen validation or test data with the frozen artifact
datadoc transform test.csv --pipeline artifacts/pipeline.json --output clean_test.csv --validate

# Generate standalone, interactive HTML audit report
datadoc report train.csv --target churn --output report.html

# Side-by-side visual comparison between raw and transformed datasets
datadoc compare train.csv clean_train.csv --target churn --html comparison.html

# Evaluate model performance against baseline
datadoc evaluate train.csv --target churn --task classification
```

### 3. Interactive Web Studio

Launch the local-first reactive dashboard:

```bash
datadoc ui train.csv --port 8000
```

Includes provenance viewer, one-click HTML report generation, and code export (`Ctrl+K` for command palette).

---

## Python SDK

DATADOC can be embedded directly into existing machine learning workflows, Airflow tasks, or inference microservices:

```python
import polars as pl
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig

# 1. Load data
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

# 2. Configure pipeline
config = PipelineConfig(
    target="churn",
    drop_identifiers=True,
    deduplicate=True,
    clip_outliers=True,
    scaling="standard",
    rare_category_min_frequency=0.01,
)

# 3. Fit strictly on training split
pipeline = DataDocPipeline(config).fit(train_df)

# 4. Transform training and unseen inference data
clean_train = pipeline.transform(train_df)
clean_test = pipeline.transform(test_df)

# 5. Serialize frozen state for production serving
pipeline.save("artifacts/pipeline.json")

# 6. Load and serve in production API
production_pipeline = DataDocPipeline.load("artifacts/pipeline.json")
scored_features = production_pipeline.transform(incoming_batch_df)
```

---

## CLI Commands Reference

| Command | Purpose | Key Flags |
| :--- | :--- | :--- |
| `datadoc health <file>` | Quick health scan with 0–100 score and quality findings | `--target` |
| `datadoc profile <file>` | Full statistical profile and column role inference | `--target`, `--explain`, `--compare <file2>` |
| `datadoc explain <file>` | AI semantic inference, missingness analysis, and feature hypotheses | `--target`, `--model`, `--recommend-config`, `--output` |
| `datadoc plan <file>` | Generates reviewable transformation plan before modifying data | `--target`, `--preset`, `--explain`, `--diff <plan2>` |
| `datadoc fit <train>` | Learns transformations from train split; serializes `pipeline.json` | `--target`, `--preset`, `--deduplicate`, `--rare-frequency` |
| `datadoc transform <file>` | Applies fitted `pipeline.json` to unseen test or inference data | `--pipeline`, `--output`, `--validate` |
| `datadoc run <file>` | One-shot end-to-end preparation with artifacts and manifests | `--target`, `--preset`, `--output-dir`, `--evaluate` |
| `datadoc report <file>` | Standalone interactive HTML health and audit report | `--target`, `--output`, `--ai`, `--ai-model`, `--open` |
| `datadoc compare <raw> <clean>` | Visual side-by-side comparison of dataset changes | `--target`, `--html`, `--json`, `--open` |
| `datadoc evaluate <file>` | Cross-validated ML benchmark (raw vs. DATADOC prepared) | `--target`, `--task`, `--ablation` |
| `datadoc export` | Exports pipeline to standalone executable Python script or Joblib | `--pipeline`, `--output`, `--format python\|joblib` |
| `datadoc lint <file>` | Audits dataset for target leakage, infinities, and invalid values | `--target` |
| `datadoc diff <f1> <f2>` | Diffs two JSON artifacts (`profile.json`, `plan.json`, `pipeline.json`) | |
| `datadoc wizard <file>` | Guided terminal assistant for interactive configuration | |
| `datadoc init` | Scaffolds a starter `datadoc.toml` configuration | `--preset` |
| `datadoc ui <file>` | Launches local-first FastAPI web studio | `--port`, `--no-browser` |
| `datadoc plugins list` | Lists all 9 registered transformation plugins and priorities | |
| `datadoc version` | Displays version and active environment details | |

---

## Architecture & Plugins

DATADOC processes tabular data through an extensible, priority-ordered transformation ladder. The sequence is mathematically ordered to ensure operations do not corrupt downstream dependencies:

```
[5: DuplicateRemover] → [10: MissingValueImputer] → [20: OutlierCapper] → [30: DatetimeDecomposer] →
[40: CategoricalEncoder] → [42: RareCategoryGrouper] → [45: StandardScaler]
```

> **Note:** `TargetEncoderPlugin` and `PolynomialFeaturesPlugin` are importable standalone utilities; they are not automatically applied by the core pipeline.

| Priority | Plugin | Action Performed |
| :---: | :--- | :--- |
| **5** | `DuplicateRemoverPlugin` | Detects duplicate records; drops them during fit to prevent distribution distortion. |
| **10** | `MissingValuePlugin` | Imputes numerical nulls with training median; categoricals with training mode. |
| **20** | `OutlierPlugin` | Calculates Tukey IQR boundaries ($Q_1 - 1.5\text{IQR}$, $Q_3 + 1.5\text{IQR}$) and caps extreme values. |
| **30** | `DatetimePlugin` | Extracts temporal features (`year`, `month`, `day`, `dayofweek`, `hour`) and cyclical sin/cos components. |
| **40** | `CategoricalEncoderPlugin` | Applies One-Hot Encoding (`drop_first=True`) to low-cardinality features; handles unseen categories. |
| **41** | `TargetEncoderPlugin` *(standalone)* | Applies Empirical Bayes smoothed target encoding: $\hat{y}_c = \frac{n \cdot \bar{y}_c + m \cdot \bar{y}}{n + m}$. *(Importable utility; not applied by the core pipeline.)* |
| **42** | `RareCategoryPlugin` | Bins infrequent categories below threshold into `__RARE__` to prevent tree overfitting. |
| **44** | `PolynomialFeaturesPlugin` *(standalone)* | Generates degree-2 interaction terms ($x_1 \cdot x_2$) and squared terms ($x^2$) for continuous features. *(Importable utility; not applied by the core pipeline.)* |
| **45** | `ScalingPlugin` | Fits and applies StandardScaler ($\frac{x - \mu}{\sigma}$) or RobustScaler ($\frac{x - Q_2}{\text{IQR}}$) to continuous inputs. |

Custom plugins can be created by subclassing `BasePlugin` and declaring entry-points under `[project.entry-points."datadoc.plugins"]`. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Configuration (`datadoc.toml`)

Pipeline settings can be committed to your repository in a `datadoc.toml` file or inside `pyproject.toml`:

```toml
[tool.datadoc]
target = "churn"
task = "classification"
preset = "balanced"
drop_identifiers = true
deduplicate = true
clip_outliers = true
categorical_threshold = 20
rare_category_min_frequency = 0.01
scaling = "standard"
strict_schema = true
```

Generate a starter configuration with:

```bash
datadoc init --preset balanced
```

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code formatting, running tests, and developing custom plugins.

Run the test suite:

```bash
python -m pytest tests/ -q
python -m ruff check datadoc/ tests/
python -m ruff format --check datadoc/ tests/
```

---

## License

DATADOC is licensed under the [MIT License](LICENSE).
