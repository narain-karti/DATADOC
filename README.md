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

**DATADOC is NOT just another EDA (Exploratory Data Analysis) tool.** It profiles data quality, lets you review a plan, fits transformations from training data, and hands you a portable artifact and Python wrapper for reuse.

### ⚡ The Impact: Why Industry Professionals Use DATADOC

Data Scientists and ML Engineers repeatedly rebuild the same preparation steps across projects.
DATADOC turns those steps into a reviewable, reusable pipeline.

- **Save boilerplate:** Review recommendations for imputing nulls, encoding categories, and optional scaling or clipping before applying them.
- **Explainable by default:** The deterministic core records roles, findings, operations, protected columns, and fitted statistics in an inspectable artifact.
- **Local-first:** The core package works offline. Optional ML, UI, and AI features are separate extras.
- **Optional AI planning:** AI can help explain or rank a constrained plan; it is never allowed to execute arbitrary generated code.
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
New here? Run the guided wizard — it asks for target + preset and runs everything:

```bash
datadoc wizard train.csv
```

Or run the one-shot happy path (profile → plan → fit → transform + manifest):

```bash
datadoc run train.csv --target churn --preset balanced --evaluate
```

Full step-by-step (auditable) workflow:

```bash
# 0. Optional: save repeatable settings (target, preset, scaling, ...)
datadoc init --preset balanced  # writes datadoc.toml

# 1. Inspect data-quality findings and column roles
datadoc profile raw_data.csv --target churn --explain --output profile.json

# 2. Review the proposed transformations before applying them
datadoc plan raw_data.csv --target churn --explain --output plan.json

# 3. Fit only on a training dataset, then save a reusable artifact
datadoc fit train.csv --target churn --preset balanced --rare-frequency 0.02 --output artifacts/churn-pipeline.json

# 4. Apply the fitted artifact to validation, test, or new data
datadoc transform validation.csv --pipeline artifacts/churn-pipeline.json --output validation-features.parquet --validate

# 5. Optionally benchmark a safe candidate pipeline against a baseline
pip install "datadoc-cli[ml]"
datadoc evaluate train.csv --target churn --task classification --ablation

# 6. Export a small executable wrapper around the fitted artifact
datadoc export --pipeline artifacts/churn-pipeline.json --output pipeline.py
# or: datadoc export --pipeline artifacts/churn-pipeline.json --format joblib --output pipeline.joblib

# 7. Lint for leakage risks / diff two plans
datadoc lint train.csv --target churn
datadoc diff plan-v1.json plan-v2.json
```

### 🖥️ Web dashboard (same pipeline, visual)

```bash
pip install "datadoc-cli[ui]"
datadoc ui train.csv --port 8000
```

The local dashboard calls the same `DataDocPipeline` behind the CLI: profile findings and roles, preparation settings (target, scaling, identifiers, dedup, clipping, cyclical datetime, rare frequency), reviewable plan, fit with output-schema preview, lineage/provenance panel, transformed-CSV download, and an executable Python export. Press `Ctrl+K`/`Cmd+K` for the command palette. Full guide: [docs/ui.html](https://narain-karti.github.io/DATADOC/ui.html).

---

## 🐍 Python SDK (Library Usage)

DATADOC is also a Python library. The stable workflow is `profile → plan → fit → transform`; the same fitted artifact can be used in notebooks, services, and batch jobs:

```python
from datadoc import DataDocPipeline, PipelineConfig
import polars as pl

# Fit only on the training split. The target is protected from feature transforms.
train_df = pl.read_csv("train.csv")
pipeline = DataDocPipeline(PipelineConfig(target="churn")).fit(train_df)
pipeline.save("artifacts/churn-pipeline.json")

# Transform data that was never used to fit statistics.
validation_df = pl.read_csv("validation.csv")
validation_features = pipeline.transform(validation_df)
```

For an observed model comparison, install the optional ML extra and call `pipeline.evaluate(train_df)` or `datadoc evaluate`. Evaluation is evidence for the declared task and split strategy; it is not a promise that cleaning always improves a model.

---

## 💻 CLI Commands Reference

| Command | Description |
|---------|-------------|
| `datadoc wizard <file>` | Guided TUI: asks target/preset/scaling, writes `datadoc.toml`, runs pipeline |
| `datadoc init` | Writes a starter `datadoc.toml` (or `pyproject.toml [tool.datadoc]`) config |
| `datadoc profile <file>` | Data-quality report + roles (`--explain`, `--compare profile2.json`) |
| `datadoc plan <file>` | Explainable plan (`--explain`, `--diff plan2.json`) |
| `datadoc fit <train>` | Learns pipeline on train only (`--preset`, `--deduplicate`, `--rare-frequency`, `--cyclical`, `--no-hour`, repeatable `--identifier-column` / `--ignore-column`) |
| `datadoc transform <file>` | Applies saved artifact (`--validate` for schema + drift checks) |
| `datadoc evaluate <file>` | Candidate vs baseline (`--ablation` for per-component deltas) |
| `datadoc export` | Wrapper for artifact (`--format python\|joblib`) |
| `datadoc run <file>` | One-shot profile→plan→fit→transform + `manifest.json` (+ `--evaluate --ablation`) |
| `datadoc lint <file>` | Leakage/pitfall lint (target duplication, nulls, infinities, duplicates) |
| `datadoc diff <a.json> <b.json>` | Diff profile/plan/pipeline artifacts |
| `datadoc plugins list` | Lists 7 registered plugins (priorities, entry-points) |
| `datadoc ui <file>` | Local FastAPI dashboard (Ctrl+K palette, lineage panel) |

Short aliases: `-t/--target`, `-o/--output`, `-p/--pipeline`, `-f/--format`.
Presets: `--preset quick|balanced|linear|tree|time|robust`. Shell completion: `datadoc --install-completion`.



---

## 🧩 Architecture & Plugins

DATADOC operates as a fitted pipeline. Every transformation learns state only from training data, saves that state to JSON, and reuses it unchanged for later datasets.

| Priority | Plugin | Action Performed |
|----------|--------|-------------|
| 5 | **DuplicateRemoverPlugin** | Detects duplicate rows; `deduplicate=True` drops them at fit (train-only) |
| 10 | **MissingValuePlugin** | Imputes missing numeric values with median, categorical with mode |
| 20 | **OutlierPlugin** | Offers optional IQR clipping; clipping is not forced by default |
| 30 | **DatetimePlugin** | Detects date strings and extracts year, month, day, day_of_week (+hour when time present, optional cyclical sin/cos) |
| 40 | **CategoricalEncoderPlugin** | Encodes categories using training vocabularies (threshold 20) and handles unseen values |
| 42 | **RareCategoryPlugin** | Groups rare categories (< `rare_category_min_frequency`) into `__RARE__` |
| 45 | **ScalingPlugin** | Applies configured standard or robust scaling, fit on training data only |

The fitted pipeline is the production source of truth. Plugin work should follow the lifecycle `analyze → recommend → apply`, with fitted state (`median`, `clip`, `vocabularies`, `rare maps`, `hour flags`, `center/spread`) serializable in `pipeline.json` (artifact v2 with `provenance`). External plugins auto-register via `datadoc.plugins` entry-points.

Want to build your own? See [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to create and register custom plugins!

---

## 🗺️ Roadmap

- [x] Core Engine with plugin orchestration
- [x] 7 Built-in deterministic plugins (duplicate, missing, outlier, datetime, encoder, rare, scaling)
- [x] Stunning Rich Terminal UI (wizard, presets, `datadoc.toml`, completion)
- [x] Pipeline export capability (`python` + `joblib`)
- [x] Polars backend and local-first pipeline artifacts (v2 + provenance)
- [x] PyPI Release (`pip install datadoc-cli`)
- [x] Constrained optional AI planning path
- [x] Session-scoped local FastAPI dashboard (Ctrl+K palette, lineage, drift)
- [x] Addictive loop: `profile --compare`, `plan --explain/--diff`, `transform --validate`, `evaluate --ablation`
- [x] Notebook widgets (`profile_to_html`, `_repr_html_`)
- [ ] Export targets for `dbt` and Apache Airflow
- [x] Local FastAPI dashboard/API companion

---

## ⚖️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions from the community! If you'd like to add a new plugin or improve the core engine, please see [CONTRIBUTING.md](CONTRIBUTING.md).

See [CHANGELOG.md](CHANGELOG.md) for the 0.5.0 release notes.
