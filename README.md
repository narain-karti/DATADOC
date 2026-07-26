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
  <a href="#installation">Install</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#cli-commands">CLI Commands</a> |
  <a href="#plugins">Plugins</a> |
  <a href="#contributing">Contributing</a>
</p>

---

## What is DATADOC?

DATADOC is an intelligent, modular **Dataset Engineering Framework** that orchestrates existing powerful libraries (Pandas, NumPy, Scikit-learn) into a standardized, reusable workflow.

**DATADOC is NOT another EDA tool.** It doesn't just show you charts. It **diagnoses**, **recommends**, and **automatically engineers** your dataset -- then hands you a portable Python script to replicate it anywhere.

### Core Philosophy

- **Orchestrate, don't replace** -- DATADOC wraps Pandas, NumPy, and Sklearn. It doesn't reinvent them.
- **Modular Plugin Architecture** -- Every transformation is an isolated, testable plugin.
- **Explainable by Default** -- Every action can be explained, rolled back, and exported as code.
- **Deterministic First** -- v1.0 uses a rule-based engine. No black-box AI decisions.

---

## Installation

```bash
# Clone and install
git clone https://github.com/narain-karti/DATADOC.git
cd DATADOC
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# Analyze your dataset's health
datadoc analyze your_data.csv

# Get recommendations without changing anything
datadoc recommend your_data.csv

# Automatically engineer your dataset
datadoc engineer your_data.csv

# Compare before vs after
datadoc compare your_data.csv

# Export a standalone Python pipeline
datadoc pipeline your_data.csv

# Generate a Markdown report
datadoc report your_data.csv

# List all available plugins
datadoc plugin
```

### Python SDK

```python
from datadoc.core.engine import DATADOC

doc = DATADOC("your_data.csv")

# Analyze
report = doc.analyze()
print(report)

# Get recommendations
for rec in doc.recommend():
    print(rec)

# Auto-engineer
clean_df = doc.engineer()
clean_df.to_csv("clean_data.csv", index=False)

# Export pipeline
with open("pipeline.py", "w") as f:
    f.write(doc.pipeline())
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `datadoc analyze <file>` | Scans dataset and shows a health report with status indicators |
| `datadoc recommend <file>` | Lists suggested engineering steps without modifying data |
| `datadoc engineer <file>` | Automatically applies all recommended transformations |
| `datadoc compare <file>` | Shows a before/after diff of the raw vs engineered dataset |
| `datadoc pipeline <file>` | Exports a standalone `.py` script with the exact Pandas code |
| `datadoc report <file>` | Generates a full Markdown report |
| `datadoc plugin` | Lists all registered plugins with priority and descriptions |
| `datadoc version` | Displays the DATADOC version |

---

## Plugins

DATADOC ships with 5 built-in plugins, executed in priority order:

| Priority | Plugin | What it Does |
|----------|--------|-------------|
| 10 | **MissingValuePlugin** | Imputes missing numeric values with median, categorical with mode |
| 20 | **OutlierPlugin** | Detects outliers via IQR and clips at 5th/95th percentiles |
| 30 | **DatetimePlugin** | Detects date columns and extracts year, month, day, day_of_week |
| 40 | **CategoricalEncoderPlugin** | One-Hot Encodes categorical columns (< 10 unique values) |
| 45 | **ScalingPlugin** | Standard scales numeric columns when scale ratio exceeds 10x |

### Plugin Interface

Every plugin implements the full `BasePlugin` interface:

```python
class BasePlugin(ABC):
    name: str              # Plugin identifier
    version: str           # Semantic version
    description: str       # One-line description
    priority: int          # Execution order (lower = first)
    supported_datatypes    # Which dtypes this plugin handles
    dependencies           # Plugins that must run before this one

    analyze(df) -> dict           # Detect issues
    recommend(analysis) -> list   # Suggest fixes
    generate_code(analysis) -> str # Export as Python code
    apply(df) -> DataFrame        # Apply transformation
    validate(df) -> bool          # Verify result
    rollback(original) -> DataFrame # Undo transformation
    explain() -> str              # Human-readable explanation
    estimate_runtime(df) -> float # Performance estimate
```

### Building Your Own Plugin

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide on building and registering custom plugins.

---

## Architecture

```
datadoc/
  core/
    engine.py          # DATADOC orchestrator class
  cli/
    app.py             # Typer CLI with Rich terminal UI
  plugins/
    base.py            # Abstract BasePlugin interface
    missing_values.py  # MissingValuePlugin
    outliers.py        # OutlierPlugin
    datetime_feat.py   # DatetimePlugin
    encoders.py        # CategoricalEncoderPlugin
    scaling.py         # ScalingPlugin
tests/
  test_core.py         # Comprehensive test suite
```

The CLI talks to the Core Engine, which orchestrates Plugins sorted by priority. Each plugin independently analyzes, recommends, and applies transformations.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check datadoc/
```

---

## Roadmap

- [x] Core Engine with plugin orchestration
- [x] 5 built-in plugins (Missing Values, Outliers, Datetime, Encoding, Scaling)
- [x] 7 CLI commands with Rich terminal UI
- [x] Pipeline export (standalone `.py` scripts)
- [x] Markdown report generation
- [x] Full plugin interface (validate, rollback, explain, estimate_runtime)
- [x] CI/CD with GitHub Actions
- [ ] PyPI release (`pip install datadoc`)
- [ ] Polars backend support
- [ ] REST API (FastAPI)
- [ ] Web Dashboard
- [ ] AI Planner (LLM replaces Rule Engine)
- [ ] Plugin Marketplace

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
