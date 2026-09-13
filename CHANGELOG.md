# Changelog

All notable changes to DATADOC are documented here. Format follows Keep a Changelog and SemVer.

## [0.5.0] - 2026-09-14
### Added
- `datadoc.toml` / `pyproject.toml [tool.datadoc]` config file support with flag overrides.
- `datadoc wizard` guided TUI for `target`, `preset`, `drop_identifiers`, `deduplicate`, `scaling`, `clip_outliers`, `rare_frequency`; writes `datadoc.toml` + run dir.
- Presets `--preset quick|balanced|linear|tree|time|robust` mapping to `scaling`, `clip_outliers`, `estimator_family`.
- Short aliases `-t` / `-o` / `-p` / `-f` for `--target` / `--output` / `--pipeline` / `--format`.
- Shell completion via Typer (`add_completion=True`) — `datadoc --install-completion`.
- `datadoc diff` for `profile`/`plan`/`pipeline` JSON comparison (incl. ops added/removed).
- `datadoc lint` leakage lint (target duplication, null target, infinities, duplicates, profile findings).
- `datadoc plugins list|show` registry introspection, `datadoc.plugins` entry-points auto-discovery.
- New deterministic plugins: `DuplicateRemoverPlugin` (P05), `RareCategoryPlugin` (P42); `CategoricalEncoderPlugin` threshold reconciled 10 → 20.
- Pipeline artifact v2 with `provenance` (schema/columns hash, rows, dedup count, version); v1 artifacts still load.
- `plan` now surfaces `deduplicate`, `outlier_clipping`, `scaling`, rare-grouping details; new `explain_plan()`.
- `fit`/`run` flags: `--deduplicate`, `--rare-frequency`, `--cyclical`, `--no-hour`.
- Repeatable `--identifier-column` / `--ignore-column` on `plan`/`fit`/`run` (also via `datadoc.toml` lists), reaching `PipelineConfig.identifier_columns` / `ignored_columns` which previously had no CLI path.
- `transform --validate` via `drift_report()` (schema + median-shift drift).
- `evaluate --ablation` via `evaluate_ablation()` (full / no_clip / no_scaling / minimal).
- Notebook widgets: `profile_to_html()` + `_repr_html_()` for Jupyter/marimo.
- `export --format python|joblib` via `export_sklearn_artifact()` (joblib needs `ml` extra).
- Dashboard: Ctrl+K/Cmd+K palette, lineage panel (`/api/pipeline/lineage`), drift endpoint, new config knobs.
- PyPI-ready wheel: `web/dist` force-included as `datadoc/_webui`, so `datadoc ui` serves the React app from a plain pip install (verified in a clean venv); `ui_server` falls back to the checkout path in dev.
- `tomli>=2.0.0` dependency on Python < 3.11 for real TOML parsing.
- New `RELEASE_CHECKLIST.md` (referenced by `docs/production.html`); `uv.lock` re-synced; `datadoc.__version__` added as the code-level version source.
### Changed
- `scikit-learn` moved back to optional `ml` extra for lighter core install.
- `PipelineConfig` docs clarified `resolved_scaling` and `plan` exposure.
- `run` now parities all `fit` flags.
### Fixed
- Removed `product_specification.md` (Retrod PMS orphan).
- Fixed `CONTRIBUTING.md` pandas/engine refs.
- `plan` no longer lets `datadoc.toml` silently override explicit `--drop-identifiers`/`--task` flags.
- `wizard` crashed when delegating to `run` (unpassed params hit raw `typer.Option` defaults); all params are now explicit.
- `diff`/`plan --diff` labels corrected to "Only in A/B" (were backwards "added/removed").
- `estimator_family="both"` now truly fits linear + tree and keeps the stronger validation score (was silently tree-only); `resolved_scaling` maps `both` → `standard`.
- Dashboard export template now emits an executable script (`__main__` + argparse) and restores `train_provenance_`.
- Commented `datadoc.toml` (including the file `datadoc init` itself writes) crashed config loading on Python 3.10 without `tomli` — the fallback parser now strips inline comments and parses lists/quoted strings.
- Fixed installed-dashboard path resolution (`datadoc/cli/_webui` → `datadoc/_webui`).
- `strict_schema` now exposed via CLI/config.
- CI re-added.

## [0.4.0] - 2026-08-09
### Added
- Leakage-safe `DataDocPipeline` (`fit`/`transform`/`save`/`load`/`evaluate`/`export_python`) in `datadoc/core/pipeline.py`.
- `PipelineConfig` with 17 knobs (`target`, `task`, `drop_identifiers`, `scaling`, `clip_outliers`, ...).
- FastAPI dashboard `datadoc ui` + `web/` Vite+React.
- Optional extras `ai` (litellm), `ml`, `ui`.
### Changed
- CLI narrowed to 7 commands `profile,plan,fit,transform,evaluate,export,run`; legacy `engine.py` removed.
- Polars as primary backend.

## [0.3.0] - 2026-07-28
### Added
- Agentic `agent.py` + `chat` command.
- FastAPI `ui` via `fastapi/uvicorn/multipart`.

## [0.2.0] - 2026-07-26
### Added
- Phase 2 AI planner `litellm`, `pydantic`, `dotenv`.
- `chat` + `engineer --ai`.

## [0.1.0] - 2026-07-26
- Initial scaffold `typer/rich/pandas` + `MissingValuePlugin`.

## Roadmap
- Export targets `dbt` / Airflow (see README).
- Text embeddings optional plugin.
