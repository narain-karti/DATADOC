<p align="center">
  <pre align="center">
 ____    _  _____  _    ____   ___   ____
|  _ \  / \|_   _|/ \  |  _ \ / _ \ / ___|
| | | |/ _ \ | | / _ \ | | | | | | | |
| |_| / ___ \| |/ ___ \| |_| | |_| | |___
|____/_/   \_\_/_/   \_\____/ \___/ \____|
  </pre>
</p>

<h3 align="center">The Open Source Operating System for Dataset Engineering.</h3>

<p align="center">
  <a href="https://narain-karti.github.io/DATADOC/"><b>&#x1F4D6; View Official Documentation Website</b></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/datadoc-cli/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/datadoc-cli.svg"></a>
  <a href="https://pypi.org/project/datadoc-cli/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/datadoc-cli.svg"></a>
  <a href="https://github.com/narain-karti/DATADOC/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="#installation"><b>Install</b></a> &bull;
  <a href="#why-datadoc"><b>Why DATADOC?</b></a> &bull;
  <a href="#quick-start"><b>Quick Start</b></a> &bull;
  <a href="#cli-commands"><b>CLI Commands</b></a> &bull;
  <a href="#architecture--plugins"><b>Architecture</b></a>
</p>

<hr>

## 🚀 What is DATADOC?

**DATADOC** is a local-first CLI and Python library for preparing tabular data for machine learning. It profiles dataset risks, creates explainable transformation plans, and saves fitted pipelines that apply the same training-derived rules to validation, test, and inference data.

Powered by **Polars**, DATADOC reads CSV and Parquet files, diagnoses missing values, identifiers, schema issues, duplicates, constants, and unsafe feature types. It does not promise model improvement: optional evaluation reports the observed result against a baseline under a reproducible split.

**DATADOC is NOT just another EDA (Exploratory Data Analysis) tool.** It doesn't just show you charts. It **fixes your data** and hands you a portable, deterministic Python script to replicate the pipeline anywhere.

### ⚡ The Impact: Why Industry Professionals Use DATADOC

Data Scientists and ML Engineers spend **80% of their time cleaning data** and only 20% training models. 
DATADOC eliminates the 80%.

- **Save Hundreds of Hours:** Stop writing boilerplate code to impute nulls, one-hot encode categorical variables, or clip outliers. DATADOC does it in one command.
- **Zero Black-Box AI:** Every transformation is strictly mathematical (IQR, medians, mode). It is 100% deterministic, explainable, and safe for enterprise production environments.
- **Lightning Fast:** By utilizing `polars` (written in Rust) instead of `pandas`, DATADOC processes millions of rows with minimal memory overhead.
- **Agentic AI Integration:** DATADOC features a built-in AI Planner and an interactive Chat Assistant that can autonomously analyze your dataset, generate engineering plans, and execute plugins using tool-calling!
- **Leakage-safe workflows:** Fitted statistics for imputation, categorical vocabularies, clipping, and scaling are learned from training data and saved as an artifact.

---

## 📦 Installation

DATADOC is published on PyPI. You can install it globally via `pip` or `uv`:

```bash
pip install datadoc-cli
```

*(Requires Python 3.10+)*

---

## 🛠️ Quick Start (CLI)

You don't need to write a single line of Python to clean your data. Just use the CLI.

```bash
# 1. Inspect data-quality findings and column roles
datadoc profile raw_data.csv --target churn --output profile.json

# 2. Review the proposed transformations before applying them
datadoc plan raw_data.csv --target churn --output plan.json

# 3. Fit only on a training dataset, then save a reusable artifact
datadoc fit train.csv --target churn --output artifacts/churn-pipeline.json

# 4. Apply the fitted artifact to validation, test, or new data
datadoc transform validation.csv --pipeline artifacts/churn-pipeline.json --output validation-features.parquet

# 5. Optionally benchmark a safe candidate pipeline against a baseline
pip install "datadoc-cli[ml]"
datadoc evaluate train.csv --target churn --task classification

# 6. Export a small executable wrapper around the fitted artifact
datadoc export --pipeline artifacts/churn-pipeline.json --output pipeline.py


```

---

## 🐍 Python SDK (Library Usage)

DATADOC is also a powerful Python library. You can import the engine directly into your Jupyter Notebooks or backend servers:

```python
from datadoc import DataDocPipeline, PipelineConfig
import polars as pl

# Fit only on the training split.
train_df = pl.read_csv("train.csv")
pipeline = DataDocPipeline(PipelineConfig(target="churn")).fit(train_df)
pipeline.save("artifacts/churn-pipeline.json")

# Transform data that was never used to fit statistics.
validation_df = pl.read_csv("validation.csv")
validation_features = pipeline.transform(validation_df)
```

---

## 💻 CLI Commands Reference

| Command | Description |
|---------|-------------|
| `datadoc profile <file>` | Produces a data-quality report and confidence-scored column roles |
| `datadoc plan <file>` | Outputs an explainable transformation plan without modifying data |
| `datadoc fit <train>` | Learns a pipeline only from training data and saves JSON state |
| `datadoc transform <file>` | Applies a saved pipeline to validation, test, or inference data |
| `datadoc evaluate <file>` | Optionally compares a safe candidate pipeline with a baseline |
| `datadoc export` | Creates an executable wrapper for a saved pipeline artifact |
| `datadoc run <file>` | Writes a profile, plan, artifact, transformed data, and lineage manifest |



---

## 🧩 Architecture & Plugins

DATADOC operates as a fitted pipeline. Every transformation learns state only from training data, saves that state to JSON, and reuses it unchanged for later datasets.

| Priority | Plugin | Action Performed |
|----------|--------|-------------|
| 10 | **MissingValuePlugin** | Imputes missing numeric values with median, categorical with mode |
| 20 | **OutlierPlugin** | Detects outliers via IQR and clips them dynamically |
| 30 | **DatetimePlugin** | Detects date strings and extracts year, month, day, day_of_week |
| 40 | **CategoricalEncoderPlugin** | One-Hot Encodes categorical columns (< 10 unique values) |
| 45 | **ScalingPlugin** | Standard scales numeric columns when scale ratio exceeds 10x |

Every plugin implements a strict `BasePlugin` interface ensuring it can `analyze()`, `apply()`, `rollback()`, and `generate_code()`. 

Want to build your own? See [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to create and register custom plugins!

---

## 🗺️ Roadmap

- [x] Core Engine with plugin orchestration
- [x] 5 Built-in deterministic plugins
- [x] Stunning Rich Terminal UI
- [x] Pipeline export capability
- [x] Polars backend and local-first pipeline artifacts
- [x] PyPI Release (`pip install datadoc-cli`)
- [x] Phase 2: Agentic AI Planner (LLM Orchestration)
- [x] Interactive AI Chat with Tool-Calling capabilities
- [ ] Export targets for `dbt` and Apache Airflow
- [x] Local FastAPI dashboard/API companion

---

## ⚖️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions from the community! If you'd like to add a new plugin or improve the core engine, please see [CONTRIBUTING.md](CONTRIBUTING.md).
