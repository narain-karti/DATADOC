<p align="center">
  <pre>
 ____    _  _____  _    ____   ___   ____
|  _ \  / \|_   _|/ \  |  _ \ / _ \ / ___|
| | | |/ _ \ | | / _ \ | | | | | | | |
| |_| / ___ \| |/ ___ \| |_| | |_| | |___
|____/_/   \_\_/_/   \_\____/ \___/ \____|
  </pre>
</p>

<h3 align="center">The Open Source Operating System for Dataset Engineering.</h3>

<p align="center">
  <a href="https://pypi.org/project/datadoc-cli/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/datadoc-cli.svg"></a>
  <a href="https://pypi.org/project/datadoc-cli/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/datadoc-cli.svg"></a>
  <a href="https://github.com/narain-karti/DATADOC/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> |
  <a href="#why-datadoc">Why DATADOC?</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#cli-commands">CLI Commands</a> |
  <a href="#architecture--plugins">Architecture</a>
</p>

---

## 🚀 What is DATADOC?

**DATADOC** is an intelligent, blazing-fast Command Line Interface (CLI) and Python Library designed to completely automate the most tedious part of Machine Learning: **Dataset Engineering and Data Cleaning**. 

Powered by a high-performance **Polars** backend, DATADOC analyzes your raw CSV files, diagnoses missing values, outliers, and schema issues, and **automatically engineers a machine-learning-ready dataset in milliseconds**.

**DATADOC is NOT just another EDA (Exploratory Data Analysis) tool.** It doesn't just show you charts. It **fixes your data** and hands you a portable, deterministic Python script to replicate the pipeline anywhere.

### ⚡ The Impact: Why Industry Professionals Use DATADOC

Data Scientists and ML Engineers spend **80% of their time cleaning data** and only 20% training models. 
DATADOC eliminates the 80%.

- **Save Hundreds of Hours:** Stop writing boilerplate code to impute nulls, one-hot encode categorical variables, or clip outliers. DATADOC does it in one command.
- **Zero Black-Box AI:** Every transformation is strictly mathematical (IQR, medians, mode). It is 100% deterministic, explainable, and safe for enterprise production environments.
- **Lightning Fast:** By utilizing `polars` (written in Rust) instead of `pandas`, DATADOC processes millions of rows with minimal memory overhead.
- **Agentic AI Integration:** DATADOC features a built-in AI Planner and an interactive Chat Assistant that can autonomously analyze your dataset, generate engineering plans, and execute plugins using tool-calling!
- **Avoid Data Leakage:** Built-in safeguards ensure that data scaling and imputation are handled correctly.

---

## 📦 Installation

DATADOC is published on PyPI. You can install it globally via `pip` or `uv`:

```bash
pip install datadoc-cli
```

*(Requires Python 3.9+)*

---

## 🛠️ Quick Start (CLI)

You don't need to write a single line of Python to clean your data. Just use the CLI.

```bash
# 1. Analyze your dataset's health (shows a beautiful terminal report)
datadoc analyze raw_data.csv

# 2. Get recommendations (DATADOC tells you exactly what is wrong)
datadoc recommend raw_data.csv

# 3. AUTO-ENGINEER! (Fixes everything and saves clean_raw_data.csv)
datadoc engineer raw_data.csv

# 4. Compare the before vs. after visually in your terminal
datadoc compare raw_data.csv clean_raw_data.csv

# 5. Export a standalone Python script to automate this in the future
datadoc pipeline raw_data.csv

# 6. Have an interactive AI session where the LLM engineers your data via chat!
datadoc chat raw_data.csv

*(Pro Tip: Add `--ai` to `analyze`, `recommend`, or `engineer` for AI-driven insights and orchestration!)*
```

---

## 🐍 Python SDK (Library Usage)

DATADOC is also a powerful Python library. You can import the engine directly into your Jupyter Notebooks or backend servers:

```python
from datadoc.core.engine import DATADOC

# Initialize the blazing-fast Polars engine
doc = DATADOC("raw_data.csv")

# Generate a diagnostic report
report = doc.analyze()
print(report)

# Automatically engineer the dataset
clean_df = doc.engineer()
clean_df.write_csv("clean_data.csv")

# Export the generated pipeline script
with open("my_pipeline.py", "w") as f:
    f.write(doc.pipeline())
```

---

## 💻 CLI Commands Reference

| Command | Description |
|---------|-------------|
| `datadoc analyze <file>` | Scans dataset and shows a health report with status indicators |
| `datadoc recommend <file>` | Lists suggested engineering steps without modifying data |
| `datadoc engineer <file>` | Automatically applies all recommended transformations |
| `datadoc chat <file>` | Starts an interactive AI session with autonomous tool-calling |
| `datadoc compare <file>` | Shows a before/after diff of the raw vs engineered dataset |
| `datadoc pipeline <file>` | Exports a standalone `.py` script with the exact Polars code |
| `datadoc visualize <file>` | Renders stunning terminal-based charts for numeric distributions |
| `datadoc plugin` | Lists all registered plugins with priority and descriptions |
| `datadoc version` | Displays the DATADOC version |

*(Note: Use the `--ai` flag on `analyze`, `recommend`, or `engineer` for LLM-powered execution!)*

---

## 🧩 Architecture & Plugins

DATADOC operates as an orchestrator. It passes your dataset through an isolated chain of plugins in a strict priority order.

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
- [x] **Polars Backend Migration (100x Performance Boost)**
- [x] PyPI Release (`pip install datadoc-cli`)
- [x] Phase 2: Agentic AI Planner (LLM Orchestration)
- [x] Interactive AI Chat with Tool-Calling capabilities
- [ ] Export targets for `dbt` and Apache Airflow
- [ ] REST API (FastAPI) wrapper

---

## ⚖️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions from the community! If you'd like to add a new plugin or improve the core engine, please see [CONTRIBUTING.md](CONTRIBUTING.md).
