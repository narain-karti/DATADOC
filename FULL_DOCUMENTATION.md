# DATADOC: The Open Source OS for Dataset Engineering

## Overview
**DATADOC** is a blazingly fast, deterministic, and highly extensible framework designed to automate the most tedious part of Machine Learning: Data Preparation. Built on top of **Polars**, it eliminates the 80% of time Data Scientists spend manually writing Pandas scripts to clean missing values, handle outliers, encode variables, and scale features.

DATADOC operates in two modes:
1. **Rule-Based Engine:** A deterministic engine that automatically detects dataset flaws and applies best-practice transformations sequentially.
2. **AI Planner Engine (Phase 2):** A semantic engine powered by **LiteLLM** that accepts natural language goals (e.g., "Clean this for a time-series forecast") and dynamically orchestrates the perfect pipeline—without ever sending your private, raw data rows to an external API.

---

## Key Features
* **Blazing Fast Backend:** Powered by Polars, processing millions of rows in milliseconds using multi-threaded Rust execution.
* **Intelligent Plugins:** Modular transformations (Missing Values, Outliers, Encoders, Scalers, Datetimes) that automatically trigger only when needed.
* **Terminal Dashboard:** A massive, rich CLI UI providing health reports, before/after diffs, and visual histograms right in your terminal.
* **Zero-Lock-in Export:** Hate black boxes? DATADOC can export the exact pipeline it generated into a standalone, highly-readable Python script.
* **Privacy-First AI:** Our AI Planner only extracts a high-level statistical metadata schema (column names, types, null counts) to prompt the LLM, keeping your actual rows secure on your local machine.
* **Bring Your Own Model:** Because we use LiteLLM, you can use any API key: Gemini, OpenAI, Claude, or local Ollama models.

---

## Installation

```bash
pip install datadoc-cli
```
*(Requires Python 3.9+)*

---

## The CLI Reference

DATADOC ships with a beautiful, Typer-based CLI.

### 1. `datadoc analyze <file.csv>`
Scans your dataset and prints a comprehensive "Health Report", checking for nulls, categorical imbalances, skewed numericals, and more.

### 2. `datadoc recommend <file.csv>`
Does not touch your data. Instead, it outputs a numbered list of suggested actions it believes should be applied based on the health scan.

### 3. `datadoc engineer <file.csv>`
The core command. It runs the dataset through the engine, applying transformations and outputs a `clean_file.csv`.
* **The AI Flag:** `datadoc engineer <file.csv> --ai --goal "Predict churn" --model "gemini/gemini-1.5-flash"`
  * Activates the AI Planner to dynamically select plugins instead of the default rule sequence.

### 4. `datadoc compare <file.csv>`
Runs the engineering pipeline in memory and prints a git-style diff comparing the "Before" vs "After" state (rows changed, columns added, missing values resolved).

### 5. `datadoc visualize <file.csv>`
Generates a massive, interactive terminal dashboard containing ASCII bar charts and histograms comparing the numerical distributions before and after cleaning.

### 6. `datadoc pipeline <file.csv>`
Exports the generated pipeline as a standalone `.py` script so you can deploy it to production, Airflow, or dbt without taking DATADOC as a dependency.

### 7. `datadoc plugin`
Lists all registered plugins, their current version, priority order, and explanations of how they work.

---

## Python SDK Reference

You can use DATADOC programmatically inside your Jupyter Notebooks or Python backends.

```python
import polars as pl
from datadoc.core.engine import DATADOC

# 1. Load the dataset
doc = DATADOC("data.csv")

# 2. Get the health report
report = doc.analyze()
print(report["rows"], report["cols"])

# 3. Engineer the data (Rule-Based)
clean_df = doc.engineer()

# 4. Engineer the data (AI Planner)
# Requires python-dotenv and a .env file with your API Key (e.g. GEMINI_API_KEY)
clean_df_ai = doc.ai_engineer(
    model="gemini/gemini-1.5-flash",
    goal="Prepare this data for XGBoost Classification"
)

# 5. Export Python Script
script = doc.pipeline()
with open("my_pipeline.py", "w") as f:
    f.write(script)
```

---

## The Plugin Architecture

DATADOC is built heavily on the **Strategy Pattern**. The core engine does almost nothing on its own; it simply orchestrates a list of `BasePlugin` objects.

### Current Built-in Plugins
1. **MissingValuePlugin:** Fills numeric nulls with the median and categorical nulls with the mode.
2. **OutlierPlugin:** Uses the Interquartile Range (IQR) method to cap extreme values at the 5th and 95th percentiles.
3. **DatetimePlugin:** Automatically detects string dates, converts them to Polars Datetime objects, and extracts features like `year`, `month`, and `day`.
4. **CategoricalEncoderPlugin:** Automatically detects low-cardinality string columns and performs one-hot encoding (dummy variables).
5. **ScalingPlugin:** Uses Min-Max scaling on numeric columns if their maximum values vary by more than 100x, ensuring algorithms like SVMs and Neural Networks converge quickly.

### Writing a Custom Plugin
Creating a new rule is as simple as inheriting from `BasePlugin`.

```python
import polars as pl
from datadoc.plugins.base import BasePlugin

class MyCustomPlugin(BasePlugin):
    name = "MyCustomPlugin"
    priority = 10 

    def analyze(self, df: pl.DataFrame) -> dict:
        # Return a dictionary of findings. 
        # If any key starts with 'has_', the engine will trigger this plugin.
        return {"has_work": True}

    def recommend(self, analysis: dict) -> list[str]:
        return ["I will apply custom logic to this dataframe."]

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        # Return the transformed dataframe
        return df.with_columns(pl.col("A") * 2)
```

Once defined, simply append it to `doc.plugins` inside `engine.py`.

---

## Privacy & The AI Planner
Security is a massive concern in enterprise data. When you run `datadoc engineer --ai`, the LLM prompt looks like this:

```json
{
  "rows": 100000,
  "columns": 5,
  "schema": {"age": "Int64", "name": "String"},
  "null_counts": {"age": 40},
  "available_plugins": ["MissingValuePlugin", "ScalingPlugin"]
}
```
**No PII or actual rows are sent over the wire.** The LLM simply replies with a strict JSON array instructing DATADOC on which local plugins to execute to achieve the user's goal.
