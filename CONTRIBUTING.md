# Contributing to DATADOC

Thank you for considering contributing to DATADOC! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/narain-karti/DATADOC.git
cd DATADOC

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest tests/ -v
```

## How to Build a Plugin

DATADOC's entire architecture is built around plugins. Every feature engineering operation is an isolated plugin that inherits from `BasePlugin`.

### Step 1: Create a new file

Create `datadoc/plugins/your_plugin.py`:

```python
import pandas as pd
from datadoc.plugins.base import BasePlugin

class YourPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "YourPlugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "What your plugin does in one sentence."

    @property
    def priority(self) -> int:
        return 50  # Lower = runs first

    def analyze(self, df: pd.DataFrame) -> dict:
        # Detect if this plugin should run
        return {"has_issue": True}

    def recommend(self, analysis_result: dict) -> list[str]:
        if analysis_result.get("has_issue"):
            return ["Description of what should be done."]
        return []

    def generate_code(self, analysis_result: dict) -> str:
        if not analysis_result.get("has_issue"):
            return ""
        return "# Your pandas code here\ndf['new_col'] = df['old_col'] * 2"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        # Your transformation logic
        return df_clean
```

### Step 2: Register the plugin

Add your plugin to `datadoc/core/engine.py` in the `self.plugins` list inside `__init__`.

### Step 3: Update the CLI

Add handling for your plugin's analysis output in the `analyze` command in `datadoc/cli/app.py`.

### Step 4: Write tests

Add tests in `tests/test_core.py` to verify your plugin's `analyze()`, `apply()`, and `recommend()` methods.

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Run `ruff check datadoc/` before submitting.
- Target Python 3.9+.

## Submitting Changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-plugin`
3. Make your changes and add tests.
4. Run the full test suite: `pytest tests/ -v`
5. Submit a pull request.

## Labels

- `good first issue` - Great for newcomers
- `plugin-idea` - Propose a new plugin
- `core-engine` - Changes to the core orchestrator
