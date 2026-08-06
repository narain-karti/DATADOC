# DATADOC: Complete Project Reading and Implementation Guide

This document explains what DATADOC was, what problems were found, what was changed, why the changes were made, how the new system works, and how developers and users should work with it.

It is written for contributors, data scientists, ML engineers, and anyone evaluating DATADOC as an open-source project.

## 1. What DATADOC is trying to become

DATADOC is a local-first data-preparation library and CLI for tabular machine-learning datasets.

Its job is not to guarantee that every automatic transformation improves every model. That guarantee would be scientifically incorrect: scaling can help a linear model and be irrelevant to a tree model; clipping an extreme value can remove real signal; dropping an identifier can remove a useful business feature; and a transformation that looks good on one holdout split can be noise.

The responsible product promise is:

> DATADOC profiles a dataset, proposes explainable transformations, learns transformation statistics from training data only, applies the resulting pipeline consistently to later data, and optionally measures the observed modeling result against a baseline.

The target users are:

- Data scientists preparing CSV or Parquet data for classification or regression.
- ML engineers who need a reusable preprocessing artifact instead of notebook-only code.
- Analysts learning predictive modeling who need visible explanations for dropped, changed, and protected columns.
- Open-source contributors who want to add deterministic, testable transformations.

The project is deliberately local-first. The core package works without an AI provider, hosted database, or hosted dashboard.

## 2. What existed before the implementation work

The original project had a useful starting shape:

```text
CSV file
  -> DATADOC engine
  -> ordered plugin list
  -> transformed Polars DataFrame
  -> CLI output / generated Python / local web UI
```

The original engine loaded a CSV file in its constructor and registered five plugins:

1. `MissingValuePlugin`
2. `OutlierPlugin`
3. `DatetimePlugin`
4. `CategoricalEncoderPlugin`
5. `ScalingPlugin`

The plugins implemented `analyze`, `recommend`, `apply`, and `generate_code`. The engine decided whether a plugin should run by checking whether an analysis dictionary contained a truthy key beginning with `has_`.

The repository also contained:

- A Typer CLI in `datadoc/cli/app.py`.
- A FastAPI local server in `datadoc/cli/ui_server.py`.
- A React/Vite dashboard in `web/`.
- A LiteLLM-based AI planner and conversational agent.
- Documentation pages under `docs/`.
- Unit tests under `tests/`.
- GitHub Actions for Python tests and linting.

That structure was enough for a prototype, but it treated preparation as a one-shot mutation rather than as a fitted, reusable data pipeline.

## 3. Problems found in the original codebase

### 3.1 The CLI engineer command was broken

The CLI called:

```python
doc.engineer(
    categorical_threshold=...,
    scaling_ratio=...,
    outlier_multiplier=...,
)
```

The engine method accepted only `progress_callback`. Therefore the main documented command failed at runtime with an unexpected-keyword-argument error.

The compatibility engine now accepts those arguments and forwards them into the fitted pipeline configuration. New users should use `fit` and `transform`; the legacy command remains available for exploratory compatibility.

### 3.2 Numeric feature columns could be misclassified as IDs

The old role detector treated a unique, evenly spaced integer column as an identifier. A valid feature such as age or an ordinal measurement could therefore be removed merely because it happened to look like a sequence.

The new profiler uses stronger evidence:

- Explicit user configuration.
- Identifier-like names such as `id`, `*_id`, `index`, `key`, or `uuid`.
- String semantics such as unique emails, names, or UUID-like values.

Numeric shape alone is no longer sufficient. Identifier detection produces a finding and does not drop the column unless the user explicitly enables identifier dropping.

### 3.3 The original pipeline was not train/test safe

The old plugins calculated medians, modes, IQR bounds, category vocabularies, and scaling statistics at the point where `apply()` was called. If a user applied the engine to all data before splitting, validation and test information could influence training transformations.

This is data leakage. It can create an unrealistically strong model score and then fail when the model sees real future data.

The new API separates fitting from transforming:

```python
pipeline.fit(train_df)
validation_features = pipeline.transform(validation_df)
test_features = pipeline.transform(test_df)
```

All learned values are stored in the fitted artifact and reused unchanged.

### 3.4 Exported code did not fully reproduce the engine

The original generated pipeline represented plugin operations but did not reliably reproduce engine-level role filtering and constant-column handling. It could therefore produce different columns from the in-memory result.

The new export is based on fitted artifact state. The generated wrapper loads the same JSON state and calls the same transformation implementation, reducing drift between runtime and exported behavior.

### 3.5 AI-generated Python was executed with `exec`

The original agent generated Python code from an LLM and executed it with `exec()`. A local dictionary is not a security sandbox. Generated code could potentially access files, environment variables, imports, or subprocesses.

That was not appropriate for a safe open-source data tool, especially if the UI or a future server exposed the path indirectly.

The new compatibility path accepts only structured plans referring to registered plugins. The production engine no longer executes LLM-generated Python. AI is optional and advisory.

### 3.6 Core dependencies were heavier than necessary

The original base installation included LiteLLM and FastAPI even when a user only wanted local Python data preparation.

Dependencies are now separated:

- Base: Polars, NumPy, Pydantic, dotenv, CLI dependencies.
- `ai`: LiteLLM.
- `ml`: scikit-learn evaluation support.
- `ui`: FastAPI, Uvicorn, and multipart support.
- `dev`: test, lint, formatting, package-build, and ML test tooling.

This keeps the core usable offline and makes the open-source package easier to install and audit.

### 3.7 The UI server allowed overly broad CORS

The original server used wildcard origins together with credentials. The server now uses configured local origins and disables credentials by default. It also stores dataset state by session identifier rather than exposing one pair of module-level variables for every request.

### 3.8 Documentation and implementation disagreed

The original documentation contained conflicting claims:

- Package version and CLI version differed.
- Some pages described Min-Max scaling while code used standard scaling.
- Some pages described percentile outlier handling while code used IQR bounds.
- Some API pages referenced methods that did not exist or did not match current behavior.
- Documentation implied that transformations automatically prevented leakage even though the old API had no train-only fit boundary.

The README, full documentation, setup instructions, and migration guidance now describe the fitted pipeline workflow.

### 3.9 Quality checks were incomplete

The original repository had good plugin tests, but the suite did not cover the CLI integration path, artifact round trips, schema enforcement, or train-only statistics. Ruff also reported unused imports and formatting issues.

New tests cover the fitted pipeline and the repository now checks formatting, linting, package building, and frontend compilation in CI.

## 4. New architecture

The central implementation is `DataDocPipeline` in `datadoc/core/pipeline.py`.

The new data flow is:

```text
Input CSV/Parquet
        |
        v
Dataset profiler
        |
        v
Column roles + findings + schema fingerprint
        |
        v
Explainable transformation plan
        |
        v
Fit on training data only
        |
        v
JSON pipeline artifact
        |
        v
Transform validation/test/inference data
        |
        v
Optional baseline evaluation and report
```

### 4.1 `PipelineConfig`

`PipelineConfig` is the user-facing configuration object. It includes:

- Target column and task type.
- Protected and ignored columns.
- Explicit identifier handling.
- Categorical threshold and missing category policy.
- Missingness indicators.
- Optional outlier clipping.
- Scaling mode: `none`, `standard`, `robust`, or `auto`.
- Estimator family: linear or tree-oriented defaults.
- Random seed and strict schema behavior.
- Optional time and group columns for evaluation.

The configuration is serialized inside the pipeline artifact so that future transformations use the same policy.

### 4.2 `DatasetProfile`

The profiler records:

- Row and column counts.
- Input schema and schema fingerprint.
- Null counts.
- Per-column cardinality.
- Duplicate row count.
- Confidence-scored column roles.
- Findings such as all-null columns, constants, identifiers, unsupported text, infinite values, duplicates, and exact target duplication.

The profiler is observational. It does not modify the input DataFrame.

### 4.3 Column roles

The profiler can classify columns as:

- `target`
- `feature_numeric`
- `feature_categorical`
- `feature_datetime`
- `identifier`
- `text`
- `ignored`
- `constant`

Every role has a confidence and a rationale. This is important for an open-source tool because users can understand and challenge an automatic decision rather than receiving an unexplained altered dataset.

### 4.4 `TransformPlan`

The plan describes what will happen before fitting. It contains operations such as:

- Drop an all-null or constant column.
- Median-impute a numeric feature.
- Add a missingness indicator.
- Encode a categorical feature.
- Convert a datetime into calendar features.
- Drop an identifier only when configured.

The plan also carries the profiler findings and protected columns.

### 4.5 Fitted state

The fitted state stores only training-derived information:

- Numeric medians.
- Missingness flags.
- Optional IQR clipping bounds.
- Categorical vocabularies.
- Frequency-encoding dictionaries.
- Datetime transformation definitions.
- Scaling centers and spreads.
- Dropped columns.
- Input and output schemas.

The state is plain JSON. This makes it inspectable, portable, versionable, and suitable for an open-source project without requiring a proprietary runtime.

## 5. Transformation behavior

### Missing values

Numeric missing values use a median learned from training data. Categorical missing values use the configured missing category. Missingness indicators can be added where null presence may itself contain signal.

All-null columns are dropped because there is no training evidence from which to construct a meaningful replacement. The drop is recorded in the profile and plan.

Infinite floating-point values are converted into missing values before numeric imputation.

### Numeric values and outliers

Outlier clipping is optional. It is not automatically applied to every numeric column because extreme values may represent valid business events.

When enabled, clipping bounds are calculated on training data only. Binary features and datetime-derived fields are excluded from scaling and are not treated as continuous measurements.

### Categorical values

Low-cardinality categories are encoded using vocabularies learned from training data. An unseen inference category produces all-zero one-hot columns rather than changing the feature schema.

Higher-cardinality categories can use frequency encoding when enabled. Target encoding is intentionally not implemented yet because it requires out-of-fold fitting and additional leakage tests.

### Datetimes

Datetime strings are parsed only when they meet the configured confidence threshold and have datetime-like naming or native date/time types. The source field is converted into stable calendar features such as year, month, day, and weekday.

Datetime-derived fields are not scaled by the general numeric scaler. This prevents accidental normalization of calendar features as if they were continuous measurements.

### Identifiers and text

Identifiers are reported, not silently destroyed by default. Users can enable `drop_identifiers` when the identifier is known to be non-predictive or unsafe.

Long text fields are reported as unsupported text rather than being silently converted into arbitrary numeric values.

## 6. Leakage-safe evaluation

Evaluation is optional and requires an explicit target.

The current evaluation path:

1. Validates that the target exists and has no missing labels.
2. Infers classification or regression when task is set to `auto`.
3. Reserves a holdout set.
4. Uses cross-validation on the remaining training data to compare a minimal baseline against the configured candidate pipeline.
5. Selects the candidate only when its cross-validation result is at least as good as the baseline.
6. Evaluates the selected choice once on the holdout set.

Classification uses balanced accuracy by default. Regression uses negative RMSE internally so larger values remain better during comparison.

The configuration also supports ordered time-based splitting and group-aware splitting. These are important because random row splitting is invalid for many time-dependent and grouped datasets.

The result includes:

- Task.
- Metric.
- Estimator family.
- Split strategy.
- Baseline score.
- Selected score.
- Improvement.
- Whether the candidate or baseline was selected.
- Feature count.
- Warnings about score uncertainty.

DATADOC does not claim that a selected transformation will universally improve production performance. It reports the observed experiment and its assumptions.

## 7. New CLI workflow

### Profile

```bash
datadoc profile data.csv --target churn --output profile.json
```

This reads the dataset without modifying it and writes a profile containing roles, null counts, findings, and schema information.

### Plan

```bash
datadoc plan data.csv --target churn --output plan.json
```

This generates a reviewable plan without applying transformations.

### Fit

```bash
datadoc fit train.csv --target churn --output artifacts/churn-pipeline.json
```

This learns transformation state from `train.csv` only.

### Transform

```bash
datadoc transform validation.csv \
  --pipeline artifacts/churn-pipeline.json \
  --output validation-features.parquet
```

The same artifact can be applied to validation, test, and future inference data. The pipeline checks required columns and compatible types before transforming.

### Evaluate

```bash
pip install "datadoc-cli[ml]"
datadoc evaluate train.csv --target churn --task classification
```

Evaluation is deliberately an optional dependency so users who only need preparation do not install machine-learning libraries.

### Export

```bash
datadoc export \
  --pipeline artifacts/churn-pipeline.json \
  --output pipeline.py
```

The exported wrapper uses the saved artifact and the same library implementation. It is not an independently reimplemented copy of the transformation rules.

### Guided run

```bash
datadoc run data.csv --target churn --task classification --output-dir runs/churn-v1
```

The run creates a profile, plan, pipeline artifact, transformed Parquet output, and manifest. With `--evaluate`, it also writes an evaluation report when ML dependencies are installed.

## 8. Python SDK workflow

```python
import polars as pl
from datadoc import DataDocPipeline, PipelineConfig

train = pl.read_csv("train.csv")
validation = pl.read_csv("validation.csv")

config = PipelineConfig(
    target="churn",
    task="classification",
    scaling="standard",
    drop_identifiers=False,
)

pipeline = DataDocPipeline(config).fit(train)
validation_features = pipeline.transform(validation)
pipeline.save("artifacts/churn-pipeline.json")

loaded = DataDocPipeline.load("artifacts/churn-pipeline.json")
same_features = loaded.transform(validation)
```

The artifact can be inspected with a text editor because it is JSON. It contains the configuration, learned state, input schema, output schema, profile, and plan.

## 9. Compatibility behavior

The original `DATADOC` facade remains available for existing users. Its behavior is now implemented through the new fitted pipeline internally.

```python
from datadoc.core.engine import DATADOC

doc = DATADOC("data.csv")
clean_df = doc.engineer()
```

This path emits a deprecation warning for supervised ML because fitting and transforming the same complete dataset is not a safe evaluation workflow.

New code should use `DataDocPipeline`.

The old `analyze`, `recommend`, `compare`, plugin listing, and generated pipeline behavior remain available for compatibility. The old AI path no longer executes generated Python.

## 10. AI behavior and security

AI is optional. The base package can run with no provider key and no network connection.

When enabled:

- The provider receives metadata rather than raw rows.
- The returned plan is checked against registered plugin names.
- The engine does not call `exec()` on AI-generated code.
- Invalid plugin names are rejected.
- Users can choose not to install the AI extra.

The local UI server now:

- Allows configured local origins instead of wildcard credentialed CORS.
- Tracks state using a session identifier.
- Validates plugin names and duplicate selections.
- Enforces declared plugin dependencies such as outlier handling after missing-value handling.

The UI is still a local companion, not a multi-tenant production service. It should not be exposed publicly without authentication, authorization, request limits, and a persistent job store.

## 11. Tests added and checks run

The existing test suite was retained and compatibility behavior was preserved. New tests cover:

- Training-only numeric medians.
- Missingness indicators.
- Unseen one-hot categories.
- Numeric feature versus identifier classification.
- Explicit identifier dropping.
- All-null column handling.
- Pipeline JSON save/load round trips.
- Missing input-column errors.
- CSV and Parquet round trips.
- Holdout evaluation reporting.

The verified checks were:

```text
95 tests passed
Ruff lint passed
Ruff format check passed
Python compilation passed
CLI profile/fit/transform/export smoke flow passed
Frontend Vite production build passed
uv build passed for source distribution and wheel
```

## 12. Packaging and open-source design

The project now treats the following as optional capabilities:

```bash
pip install datadoc-cli          # core CLI and preparation
pip install "datadoc-cli[ml]"   # evaluation support
pip install "datadoc-cli[ai]"   # optional AI planning
pip install "datadoc-cli[ui]"   # local FastAPI dashboard
pip install "datadoc-cli[dev]"  # contributor tooling
```

This structure reduces installation friction and lets contributors test the core without requiring a commercial model provider.

The project uses plain JSON artifacts and deterministic seeds where applicable. This improves reviewability, portability, bug reports, and reproducibility across operating systems.

## 13. What is intentionally not finished yet

The implementation establishes a safe and usable v1 foundation. The following are deliberate next steps rather than hidden guarantees:

1. A formal external plugin lifecycle with public `fit`, `transform`, and serialized plugin-state interfaces.
2. Candidate search across multiple safe strategies such as no clipping versus clipping, standard versus robust scaling, and rare-category grouping.
3. Out-of-fold target encoding with dedicated leakage tests.
4. A full dashboard redesign around profile, plan approval, artifact management, and evaluation reports.
5. Richer reports with confidence intervals, fold-level scores, feature lineage, and model-family comparisons.
6. Better support for text, joins, geospatial fields, and distributed datasets.
7. A stable plugin registry and contributor documentation for third-party transformations.

These should be added incrementally. The project should not trade away leakage safety or explainability to advertise a larger feature list.

## 14. Recommended contributor workflow

1. Read `README.md`, this guide, `MIGRATION.md`, and `CONTRIBUTING.md`.
2. Add or update a focused test before changing transformation behavior.
3. Keep transformation state JSON-serializable.
4. Ensure all statistics are learned in `fit`, never in `transform`.
5. Add a schema, leakage, and artifact round-trip test for new behavior.
6. Run the local checks:

   ```bash
   pytest -q
   ruff check datadoc tests
   ruff format --check datadoc tests
   python -m compileall -q datadoc
   ```

7. Update README and migration documentation when public behavior changes.
8. Keep AI/provider integrations optional and never execute untrusted generated code.

## 15. Practical model-training recommendation

Use DATADOC as a preparation and measurement component in a larger modeling workflow:

```text
Raw data
  -> define target, time/group rules, and protected columns
  -> reserve a final test set
  -> fit DATADOC on training rows
  -> transform validation/test rows with the fitted artifact
  -> train the model
  -> evaluate on untouched test data
  -> retain the simplest pipeline with acceptable validated performance
```

A clean dataset is not automatically a good dataset. The best result is a pipeline that is reproducible, leakage-safe, understandable, and empirically useful for the user’s declared model objective.
