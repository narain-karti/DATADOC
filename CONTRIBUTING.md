# Contributing to DATADOC

Thank you for considering contributing to DATADOC! This guide reflects the current **0.4.0+ leakage-safe fitted pipeline** architecture.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/narain-karti/DATADOC.git
cd DATADOC

# Install in development mode (core + dev)
pip install -e ".[dev]"

# Or install everything (ml + ui + ai)
pip install -e ".[all]"

# Run the test suite
pytest tests/ -v
ruff check datadoc/
```

## Architecture at a Glance

- **Core engine:** `datadoc/core/pipeline.py` — `DataDocPipeline(PipelineConfig)` with `profile() -> plan() -> fit() -> transform()` and `save()/load()` artifact (`pipeline.json`).
- **Plugins:** `datadoc/plugins/*.py` inherit `BasePlugin` (`datadoc/plugins/base.py:5`). They are **deterministic, stateless advisors**; the fitted pipeline in `pipeline.py:532` is the production source of truth (frozen `state_` learned only from train).
- **CLI:** `datadoc/cli/app.py` (Typer + Rich). Commands `profile, plan, fit, transform, evaluate, export, run, ui, wizard, diff, lint, plugins`.
- **Polars:** All DataFrames are `polars.DataFrame`. We use `polars>=0.20.0` for Rust speed and Arrow zero-copy.

## How to Build a Plugin

### Step 1: Create a new file

Create `datadoc/plugins/your_plugin.py`:

```python
import polars as pl
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
        return 50  # Lower = runs first (10=missing, 20=outlier, 30=datetime, 40=encoder, 45=scaling)

    def analyze(self, df: pl.DataFrame) -> dict:
        # Detect if this plugin should run
        return {"has_issue": True, "columns": []}

    def recommend(self, analysis_result: dict) -> list[str]:
        if analysis_result.get("has_issue"):
            return ["Description of what should be done."]
        return []

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        df_clean = df.clone()
        # Your Polars transformation logic (no fit here — keep stateless)
        return df_clean
```

Key lifecycle: `analyze(df) -> recommend(result) -> apply(df)`. For **fitted** work, add state to `DataDocPipeline.fit()` in `datadoc/core/pipeline.py:396` and ` _transform_with_state():532` — see existing `numeric/categorical/datetime/scaling` examples.

### Step 2: Auto-register via entry points (recommended)

No manual list edit needed. The registry auto-discovers:

1. Built-ins listed in `datadoc/plugins/registry.py:BUILTIN_PLUGINS`.
2. External plugins via `pyproject.toml`:

```toml
[project.entry-points."datadoc.plugins"]
my_plugin = "my_package.my_plugin:MyPlugin"
```

Run `datadoc plugins list` to verify discovery. Priorities are sorted automatically.

Legacy: you can also append to `BUILTIN_PLUGINS` for local development.

### Step 3: Wire into PipelineConfig (if needed)

If your plugin needs knobs, add fields to `PipelineConfig` (`datadoc/core/pipeline.py:60`) and thread them through `app.py:_pipeline_config()`. Keep defaults deterministic.

### Step 4: Write tests

Add tests in `tests/test_core.py` and `tests/test_pipeline.py`:

```python
def test_your_plugin_analyze():
    df = pl.DataFrame({"a": [1, None, 3]})
    p = YourPlugin()
    assert p.analyze(df)["has_issue"]
    assert p.priority == 50
```

Cover `analyze()`, `recommend()`, `apply()`, and leakage-safe `fit->transform` with train/test split.

## CLI Conventions

- Prefer `--preset linear|tree|time` over long flag combos; `datadoc.toml` / `pyproject.toml [tool.datadoc]` stores `target`, `task`, `drop_identifiers`, etc.
- Keep commands composable: `datadoc profile` and `plan` are read-only; `fit` learns state; `transform` reapplies it.
- `datadoc run` is the one-shot happy path; keep `fit` parity with `run`.
- Use `add_completion=True` and rich help panels — every command shows examples.

## Code Style

- Ruff `line-length=100`, target `py310` (`pyproject.toml:65`).
- `ruff check datadoc/ && ruff format datadoc/`
- No `pandas` in new code — use `polars` idioms (`pl.col(...).fill_null(...)`, `to_dummies`, `dt.year()`).

## Submitting Changes

1. Fork → branch `feature/your-plugin` → add tests.
2. `pytest tests/ -v` and `python -m datadoc.cli.app --help` must pass.
3. PR description must include before/after `plan` JSON diff if you touched the pipeline.

## Labels

- `good first issue` - Great for newcomers
- `plugin-idea` - Propose a new plugin
- `core-engine` - Changes to `pipeline.py`
- `cli-dx` - CLI / wizard / config work
