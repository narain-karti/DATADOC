# DATADOC: Complete User and Developer Guide

DATADOC is a local-first open-source toolkit for preparing tabular datasets for supervised machine learning. It profiles a dataset, explains possible data-quality problems, creates a transformation plan, fits that plan only on training data, and reuses the fitted rules for validation, test, and inference data.

The central idea is simple:

```text
profile → plan → fit on train → transform validation/test/new data → evaluate
```

DATADOC does not promise that cleaning always improves model performance. When a target is supplied and the optional ML dependency is installed, it measures the result against a reproducible baseline and can honestly report that no improvement was found.

For the implementation explanation, read [PROJECT_IMPLEMENTATION_GUIDE.md](PROJECT_IMPLEMENTATION_GUIDE.md). For migration from the original mutable API, read [MIGRATION.md](MIGRATION.md). The visual documentation site is in [`docs/`](docs/) and begins at [`docs/index.html`](docs/index.html).

For maintainers, [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) describes the CI, artifact, Git tag, and PyPI upload steps.

## 1. Installation

The core package is intentionally small and works without an AI provider, web server, or scikit-learn:

```bash
pip install datadoc-cli
```

Optional capabilities are installed only when needed:

```bash
pip install "datadoc-cli[ml]"   # model benchmarking
pip install "datadoc-cli[ui]"   # local FastAPI dashboard
pip install "datadoc-cli[ai]"   # constrained advisory AI planner
pip install "datadoc-cli[dev]"  # contributor tooling
```

DATADOC supports Python 3.10, 3.11, and 3.12. Input and output formats are CSV and Parquet.

## 2. The mental model

Imagine a dataset with a numeric `age`, a categorical `plan`, a timestamp, missing values, and a target called `churn`.

1. `profile` looks at the columns and reports facts: null counts, cardinality, duplicates, constants, suspicious identifiers, parse confidence, and warnings.
2. `plan` turns those facts into proposed operations. It does not modify the dataset.
3. `fit` learns reusable values from training data: numeric medians, category vocabularies, missing-value indicators, outlier bounds when enabled, and scaling centers/spreads.
4. `transform` applies those frozen values to another dataset. It never recalculates them from validation, test, or inference rows.
5. `evaluate` can compare a minimal baseline with a candidate pipeline for a declared classification or regression task.

This separation prevents data leakage. For example, a validation row cannot change the median used to fill a training feature, and a category appearing only in validation cannot enlarge the training feature schema.

## 3. The recommended CLI workflow

### Step 1: profile

```bash
datadoc profile data.csv --target churn --output profile.json
```

This is read-only. A declared target is protected and is never treated as a feature. The output contains schema, null counts, cardinality, confidence-scored roles, findings, duplicate-row count, and a schema fingerprint.

### Step 2: review a plan

```bash
datadoc plan data.csv --target churn --task classification --output plan.json
```

The plan lists operations such as numeric median imputation, categorical missing-value handling, one-hot encoding, optional missingness indicators, and optional scaling. Identifier suggestions are findings; they are not silently dropped.

### Step 3: fit on training data

```bash
datadoc fit train.csv \
  --target churn \
  --task classification \
  --output artifacts/churn-pipeline.json
```

Only `train.csv` is used to learn pipeline state. The JSON artifact includes the input schema, output schema, configuration, profile, plan, artifact version, and fitted state.

### Step 4: transform later data

```bash
datadoc transform validation.csv \
  --pipeline artifacts/churn-pipeline.json \
  --output validation-features.parquet

datadoc transform test.csv \
  --pipeline artifacts/churn-pipeline.json \
  --output test-features.parquet
```

The pipeline checks compatibility before transforming. Missing required columns or incompatible artifact versions produce actionable errors. Unseen categories are handled without changing the output schema.

### Step 5: evaluate when a target exists

```bash
pip install "datadoc-cli[ml]"
datadoc evaluate train.csv \
  --target churn \
  --task classification \
  --estimator linear \
  --output evaluation.json
```

Classification uses balanced accuracy as the primary comparison. Regression uses RMSE. The report names the estimator family, metric, split strategy, baseline score, selected score, improvement, selected pipeline, feature count, and warnings. Use `--time-column` for ordered evaluation and `--group-column` when rows must stay grouped.

For benchmarking, DATADOC passes numeric transformed features to the estimator. Retained text or identifier columns remain visible in the prepared output but are excluded from the model matrix with a warning. Any remaining numeric missing values are filled with medians learned from the current training fold, never from the validation fold.

### One-command run

```bash
datadoc run data.csv \
  --target churn \
  --task classification \
  --output-dir runs/churn-v1 \
  --evaluate
```

This creates `profile.json`, `plan.json`, `pipeline.json`, `transformed.parquet`, and `manifest.json`; `evaluation.json` is added when evaluation is requested and the ML extra is available. For a production model decision, keep an untouched external test set and do not treat the convenience run as a replacement for a deliberate train/validation/test split.

### Export code

```bash
datadoc export \
  --pipeline artifacts/churn-pipeline.json \
  --output pipeline.py
```

The export is a small executable wrapper that loads the same artifact and calls the same runtime implementation. It is intended for transparent deployment and review; it does not duplicate or reinterpret fitted statistics.

## 4. Python SDK

```python
import polars as pl
from datadoc import DataDocPipeline, PipelineConfig

train = pl.read_csv("train.csv")
validation = pl.read_csv("validation.csv")

config = PipelineConfig(
    target="churn",
    task="classification",
    scaling="standard",
    clip_outliers=False,
    random_seed=42,
)

pipeline = DataDocPipeline(config)
profile = pipeline.profile(train)
plan = pipeline.plan(train)
pipeline.fit(train)

validation_features = pipeline.transform(validation)
pipeline.save("artifacts/churn-pipeline.json")

loaded = DataDocPipeline.load("artifacts/churn-pipeline.json")
same_features = loaded.transform(validation)
assert same_features.columns == validation_features.columns

# Optional model-oriented evidence:
report = DataDocPipeline(config).evaluate(train, target="churn")
print(report.to_dict())
```

Use `profile` and `plan` before `fit` when you want to inspect decisions. Use `load` in a separate inference process. The target remains in the transformed frame so callers can separate labels from features explicitly.

## 5. What is protected and what is configurable?

- A declared target is not imputed, encoded, scaled, clipped, or dropped as a feature.
- Numeric missing values use a training median; categorical missing values use a configured marker.
- Invalid numeric infinities become missing before imputation.
- Missingness indicators can preserve signal carried by the fact that a value was absent.
- One-hot vocabularies are learned on training data; unseen inference categories are safe.
- High-cardinality strings are reported and can use frequency encoding; they are not silently deleted.
- Outlier clipping is opt-in and is not applied to binary, target, identifier, or datetime-derived fields.
- Scaling is configurable and defaults according to estimator family; tree-style models do not need it.
- Date-like fields are parsed only when confidence is sufficient; parse warnings remain visible.
- Numeric uniqueness alone never proves that a column is an identifier. Name and value evidence are combined.
- Unsupported text, joins, images, geospatial features, and distributed processing are outside the stable tabular scope and are reported rather than guessed.

## 6. Legacy compatibility

The original `DATADOC(file).analyze()`, `.recommend()`, `.engineer()`, `.compare()`, `.visualize()`, `.report()`, and `.pipeline()` commands remain available as compatibility paths. New work should use `DataDocPipeline`.

`engineer()` is deprecated for production use because fitting and transforming in one mutable operation makes train/test boundaries easy to misuse. It now warns that it cannot make a model-performance claim without an explicit target and validation setup. The old arbitrary-code execution path is not part of the production workflow.

## 7. Local web UI

Install the UI extra and start the dashboard with a local file:

```bash
pip install "datadoc-cli[ui]"
datadoc ui data.csv
```

The UI consumes the same profile, plan, fit, transform-preview, and code-export concepts as the CLI. It uses session-scoped state, binds locally by default, restricts CORS to configured local origins, and does not execute AI-generated code. The frontend is in `web/src/App.jsx`; build it with `npm ci` and `npm run build` from `web/`.

## 8. Open-source development

Install contributor dependencies and run the checks:

```bash
pip install -e ".[dev,ml,ui]"
pytest -q
ruff check datadoc tests
ruff format --check datadoc tests
python -m compileall -q datadoc
python -m build
```

Transformation contributions must prove four things: learned values come from training data only, the transform is stable for unseen or missing values, the artifact round-trips, and the generated wrapper agrees with runtime output. Prefer small focused pull requests, document behavior changes, and keep optional dependencies optional.

## 9. Current boundaries and future work

The stable product is tabular classification and regression preparation. Future work can add out-of-fold target encoding, richer repeated cross-validation, calibrated uncertainty, stronger artifact schemas, text and geospatial adapters, and distributed execution. Those additions should preserve the same safety contract rather than make automation more aggressive.

## 10. License

DATADOC is licensed under the MIT License. See [LICENSE](LICENSE).
