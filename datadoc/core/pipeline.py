"""Leakage-safe, serializable preparation pipelines for tabular data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

import polars as pl


ARTIFACT_VERSION = 1
ID_NAME_PATTERN = re.compile(r"(?:^|[_\-\s])(id|index|key|uuid)$", re.IGNORECASE)
DATETIME_NAME_PATTERN = re.compile(r"(?:date|time|timestamp|_at)$", re.IGNORECASE)


class DataDocError(ValueError):
    """Raised when a dataset or pipeline artifact is not compatible."""


@dataclass
class ColumnRole:
    name: str
    role: Literal[
        "target",
        "feature_numeric",
        "feature_categorical",
        "feature_datetime",
        "identifier",
        "text",
        "ignored",
        "constant",
    ]
    confidence: float
    rationale: str


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    schema: dict[str, str]
    null_counts: dict[str, int]
    cardinality: dict[str, int]
    roles: list[ColumnRole]
    findings: list[dict[str, Any]]
    duplicate_rows: int
    schema_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    target: str | None = None
    task: Literal["auto", "classification", "regression"] = "auto"
    protected_columns: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)
    drop_identifiers: bool = False
    categorical_threshold: int = 20
    datetime_parse_threshold: float = 0.9
    add_missing_indicators: bool = True
    categorical_missing_value: str = "__MISSING__"
    rare_category_min_frequency: float = 0.0
    encode_high_cardinality: bool = True
    clip_outliers: bool = False
    outlier_multiplier: float = 1.5
    scaling: Literal["none", "standard", "robust", "auto"] = "auto"
    estimator_family: Literal["linear", "tree", "both"] = "linear"
    random_seed: int = 42
    strict_schema: bool = True
    time_column: str | None = None
    group_column: str | None = None

    def resolved_scaling(self) -> Literal["none", "standard", "robust"]:
        if self.scaling != "auto":
            return self.scaling
        return "standard" if self.estimator_family == "linear" else "none"


@dataclass
class TransformPlan:
    operations: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    protected_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    task: str
    metric: str
    estimator_family: str
    split_strategy: str
    baseline_score: float
    selected_score: float
    improvement: float
    selected_pipeline: str
    feature_count: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_dataset(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _normalize_numeric_missing(pl.read_csv(path, infer_schema_length=10_000))
    if suffix in {".parquet", ".pq"}:
        return _normalize_numeric_missing(pl.read_parquet(path))
    raise DataDocError("Only CSV and Parquet files are supported.")


def write_dataset(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
        return
    if suffix in {".parquet", ".pq"}:
        df.write_parquet(path)
        return
    raise DataDocError("Output must use a .csv or .parquet extension.")


def _schema_fingerprint(df: pl.DataFrame) -> str:
    schema = [(name, str(dtype)) for name, dtype in df.schema.items()]
    return hashlib.sha256(json.dumps(schema).encode("utf-8")).hexdigest()[:16]





def _normalize_numeric_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Represent floating-point NaN values as nulls throughout the pipeline."""
    expressions = [
        pl.when(pl.col(name).is_nan()).then(None).otherwise(pl.col(name)).alias(name)
        for name, dtype in df.schema.items()
        if dtype.is_float()
    ]
    return df.with_columns(expressions) if expressions else df


def _datetime_ratio(series: pl.Series) -> float:
    if series.dtype != pl.String or series.len() == 0:
        return 0.0
    try:
        parsed = series.cast(pl.String).str.to_datetime(strict=False)
    except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
        return 0.0
    return (parsed.len() - parsed.null_count()) / max(series.len(), 1)


def profile_dataset(df: pl.DataFrame, config: PipelineConfig | None = None) -> DatasetProfile:
    df = _normalize_numeric_missing(df)
    config = config or PipelineConfig()
    roles: list[ColumnRole] = []
    findings: list[dict[str, Any]] = []
    null_counts: dict[str, int] = {}
    cardinality: dict[str, int] = {}

    for name in df.columns:
        series = df[name]
        non_null = series.drop_nulls()
        null_count = series.null_count()
        unique = non_null.n_unique() if non_null.len() else 0
        null_counts[name] = null_count
        cardinality[name] = unique

        if name == config.target:
            roles.append(ColumnRole(name, "target", 1.0, "Explicitly declared target column."))
            continue
        if name in config.protected_columns:
            roles.append(ColumnRole(name, "ignored", 1.0, "User-protected column."))
            continue
        if name in config.ignored_columns:
            roles.append(ColumnRole(name, "ignored", 1.0, "User-ignored column."))
            continue
        if non_null.len() == 0:
            roles.append(ColumnRole(name, "constant", 1.0, "Column contains only null values."))
            findings.append(
                {
                    "severity": "warning",
                    "column": name,
                    "code": "all_null",
                    "message": "Column will be dropped because it contains no usable values.",
                }
            )
            continue
        if unique <= 1:
            roles.append(ColumnRole(name, "constant", 1.0, "Column has zero variance."))
            findings.append(
                {
                    "severity": "info",
                    "column": name,
                    "code": "constant",
                    "message": "Column has zero variance.",
                }
            )
            continue
        if name in config.identifier_columns or (
            series.dtype.is_integer() and ID_NAME_PATTERN.search(name)
        ):
            roles.append(
                ColumnRole(name, "identifier", 0.95, "Name and type indicate identifier semantics.")
            )
            findings.append(
                {
                    "severity": "warning",
                    "column": name,
                    "code": "identifier",
                    "message": "Identifier-like feature is retained unless drop_identifiers is enabled.",
                }
            )
            continue
        if series.dtype.is_numeric():
            if series.dtype.is_float() and series.is_infinite().any():
                findings.append(
                    {
                        "severity": "warning",
                        "column": name,
                        "code": "infinite_values",
                        "message": "Infinite values will be treated as missing values.",
                    }
                )
            roles.append(ColumnRole(name, "feature_numeric", 1.0, "Numeric, non-constant feature."))
            continue
        if series.dtype in (pl.Date, pl.Datetime):
            roles.append(
                ColumnRole(name, "feature_datetime", 1.0, "Native Polars date/time column.")
            )
            continue
        if series.dtype == pl.String:
            parse_ratio = _datetime_ratio(series)
            if parse_ratio >= config.datetime_parse_threshold and DATETIME_NAME_PATTERN.search(
                name
            ):
                roles.append(
                    ColumnRole(
                        name,
                        "feature_datetime",
                        parse_ratio,
                        "Datetime parsing passed configured confidence threshold.",
                    )
                )
                continue
            average_length = float(non_null.cast(pl.String).str.len_chars().mean() or 0)
            if unique == non_null.len() and (
                "email" in name.lower() or "name" in name.lower() or "uuid" in name.lower()
            ):
                roles.append(
                    ColumnRole(name, "identifier", 0.9, "Unique string with identifier-like name.")
                )
                findings.append(
                    {
                        "severity": "warning",
                        "column": name,
                        "code": "identifier",
                        "message": "Identifier-like feature is retained unless drop_identifiers is enabled.",
                    }
                )
            elif average_length > 80:
                roles.append(
                    ColumnRole(
                        name,
                        "text",
                        0.8,
                        "Long free-text values require a text-specific transformer.",
                    )
                )
                findings.append(
                    {
                        "severity": "warning",
                        "column": name,
                        "code": "unsupported_text",
                        "message": "Free-text feature is not automatically encoded.",
                    }
                )
            else:
                roles.append(
                    ColumnRole(
                        name,
                        "feature_categorical",
                        0.9,
                        "String feature suitable for categorical handling.",
                    )
                )
            continue
        roles.append(ColumnRole(name, "ignored", 0.5, f"Unsupported datatype: {series.dtype}."))

    duplicate_rows = int(df.is_duplicated().sum()) if df.height else 0
    if duplicate_rows:
        findings.append(
            {
                "severity": "warning",
                "column": None,
                "code": "duplicate_rows",
                "message": f"{duplicate_rows} duplicate rows detected; they are not dropped automatically.",
            }
        )

    if config.target and config.target in df.columns:
        for name in df.columns:
            if name != config.target and df[name].equals(df[config.target]):
                findings.append(
                    {
                        "severity": "error",
                        "column": name,
                        "code": "target_leakage",
                        "message": f"Column exactly duplicates target '{config.target}'.",
                    }
                )

    return DatasetProfile(
        rows=df.height,
        columns=df.width,
        schema={name: str(dtype) for name, dtype in df.schema.items()},
        null_counts=null_counts,
        cardinality=cardinality,
        roles=roles,
        findings=findings,
        duplicate_rows=duplicate_rows,
        schema_fingerprint=_schema_fingerprint(df),
    )


class DataDocPipeline:
    """A fitted train-only transformation pipeline with a JSON artifact."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.profile_: DatasetProfile | None = None
        self.plan_: TransformPlan | None = None
        self.state_: dict[str, Any] = {}
        self.input_schema_: dict[str, str] = {}
        self.output_schema_: dict[str, str] = {}
        self.fitted_ = False

    def profile(self, df: pl.DataFrame) -> DatasetProfile:
        self.profile_ = profile_dataset(df, self.config)
        return self.profile_

    def plan(self, df: pl.DataFrame) -> TransformPlan:
        profile = self.profile(df)
        operations: list[dict[str, Any]] = []
        for role in profile.roles:
            if role.role == "constant":
                operations.append(
                    {"operation": "drop", "column": role.name, "reason": role.rationale}
                )
            elif role.role == "identifier" and self.config.drop_identifiers:
                operations.append(
                    {
                        "operation": "drop",
                        "column": role.name,
                        "reason": "Identifier dropping explicitly enabled.",
                    }
                )
            elif role.role == "feature_numeric":
                operations.append(
                    {
                        "operation": "numeric_imputation",
                        "column": role.name,
                        "reason": "Median learned from training data.",
                    }
                )
            elif role.role == "feature_categorical":
                operations.append(
                    {
                        "operation": "categorical_encoding",
                        "column": role.name,
                        "reason": "Vocabulary learned from training data.",
                    }
                )
            elif role.role == "feature_datetime":
                operations.append(
                    {
                        "operation": "datetime_features",
                        "column": role.name,
                        "reason": "Calendar features from parsed timestamps.",
                    }
                )
        self.plan_ = TransformPlan(operations, profile.findings, self.config.protected_columns)
        return self.plan_

    def fit(self, train_df: pl.DataFrame, target: str | None = None) -> "DataDocPipeline":
        train_df = _normalize_numeric_missing(train_df)
        if target is not None:
            self.config.target = target
        if self.config.target and self.config.target not in train_df.columns:
            raise DataDocError(
                f"Target column '{self.config.target}' is not present in training data."
            )

        profile = self.profile(train_df)
        self.plan(train_df)
        self.input_schema_ = {name: str(dtype) for name, dtype in train_df.schema.items()}
        role_map = {role.name: role for role in profile.roles}
        state: dict[str, Any] = {
            "dropped": [],
            "numeric": {},
            "categorical": {},
            "datetime": {},
            "scaling": {},
        }

        for name, role in role_map.items():
            if role.role == "target" or role.role == "ignored":
                continue
            series = train_df[name]
            if role.role == "constant" or (
                role.role == "identifier" and self.config.drop_identifiers
            ):
                state["dropped"].append(name)
            elif role.role == "feature_numeric":
                prepared = (
                    series.replace(float("inf"), None).replace(float("-inf"), None)
                    if series.dtype.is_float()
                    else series
                )
                median = prepared.median()
                if median is None:
                    state["dropped"].append(name)
                    continue
                values = prepared.drop_nulls()
                numeric_state: dict[str, Any] = {
                    "median": float(median),
                    "missing": prepared.null_count() > 0,
                }
                if self.config.clip_outliers and values.len() >= 4:
                    q1, q3 = values.quantile(0.25), values.quantile(0.75)
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        numeric_state["clip"] = [
                            float(q1 - self.config.outlier_multiplier * iqr),
                            float(q3 + self.config.outlier_multiplier * iqr),
                        ]
                state["numeric"][name] = numeric_state
            elif role.role == "feature_categorical":
                values = series.cast(pl.String).fill_null(self.config.categorical_missing_value)
                counts = values.value_counts()
                frequencies = {str(row[0]): int(row[1]) for row in counts.iter_rows()}
                total = max(values.len(), 1)
                kept = sorted(
                    value
                    for value, count in frequencies.items()
                    if count / total >= self.config.rare_category_min_frequency
                )
                if len(kept) <= self.config.categorical_threshold:
                    state["categorical"][name] = {
                        "kind": "one_hot",
                        "categories": kept,
                        "missing": series.null_count() > 0,
                    }
                elif self.config.encode_high_cardinality:
                    state["categorical"][name] = {
                        "kind": "frequency",
                        "frequencies": {key: count / total for key, count in frequencies.items()},
                        "missing": series.null_count() > 0,
                    }
                else:
                    state["dropped"].append(name)
            elif role.role == "feature_datetime":
                state["datetime"][name] = {"source_dtype": str(series.dtype)}

        transformed = self._transform_with_state(train_df, state, validate_schema=False)
        scaling = self.config.resolved_scaling()
        if scaling != "none":
            datetime_feature_prefixes = tuple(f"{name}__" for name in state["datetime"])
            for name in transformed.columns:
                if name == self.config.target or not transformed[name].dtype.is_numeric():
                    continue
                if name.startswith(datetime_feature_prefixes):
                    continue
                values = transformed[name].drop_nulls()
                if values.n_unique() <= 2:
                    continue
                if scaling == "standard":
                    center, spread = values.mean(), values.std()
                else:
                    center = values.median()
                    spread = values.quantile(0.75) - values.quantile(0.25)
                if center is not None and spread not in (None, 0):
                    state["scaling"][name] = {
                        "kind": scaling,
                        "center": float(center),
                        "spread": float(spread),
                    }

        transformed = self._transform_with_state(train_df, state, validate_schema=False)
        self.state_ = state
        self.output_schema_ = {name: str(dtype) for name, dtype in transformed.schema.items()}
        self.fitted_ = True
        return self

    def _validate_input_schema(self, df: pl.DataFrame) -> None:
        expected = (
            set(self.input_schema_) - {self.config.target}
            if self.config.target
            else set(self.input_schema_)
        )
        missing = sorted(expected - set(df.columns))
        if missing:
            raise DataDocError(f"Input is missing required columns: {', '.join(missing)}")
        if self.config.strict_schema:
            incompatible = []
            for name in expected:
                actual = df.schema[name]
                expected_type = self.input_schema_[name]
                if actual == pl.Null or str(actual) == expected_type:
                    continue
                if actual.is_numeric() and any(
                    token in expected_type for token in ("Int", "UInt", "Float", "Decimal")
                ):
                    continue
                incompatible.append(name)
            if incompatible:
                raise DataDocError(
                    f"Input has incompatible datatypes for: {', '.join(incompatible)}"
                )

    def _transform_with_state(
        self, df: pl.DataFrame, state: dict[str, Any], validate_schema: bool = True
    ) -> pl.DataFrame:
        df = _normalize_numeric_missing(df)
        if validate_schema:
            self._validate_input_schema(df)
        output = df.clone()
        target = self.config.target
        drop_columns = [
            name for name in state["dropped"] if name in output.columns and name != target
        ]
        if drop_columns:
            output = output.drop(drop_columns)

        for name, spec in state["numeric"].items():
            if name not in output.columns or name == target:
                continue
            expr = pl.col(name)
            if output[name].dtype.is_float():
                expr = pl.when(expr.is_infinite()).then(None).otherwise(expr)
            if self.config.add_missing_indicators and spec["missing"]:
                output = output.with_columns(
                    pl.col(name).is_null().cast(pl.UInt8).alias(f"{name}__missing")
                )
            expr = expr.fill_null(spec["median"])
            if "clip" in spec:
                expr = expr.clip(spec["clip"][0], spec["clip"][1])
            output = output.with_columns(expr.alias(name))

        for name, spec in state["datetime"].items():
            if name not in output.columns or name == target:
                continue
            if output[name].dtype == pl.String:
                output = output.with_columns(pl.col(name).str.to_datetime(strict=False).alias(name))
            if output[name].dtype not in (pl.Date, pl.Datetime):
                raise DataDocError(f"Datetime column '{name}' could not be parsed.")
            output = output.with_columns(
                [
                    pl.col(name).dt.year().alias(f"{name}__year"),
                    pl.col(name).dt.month().alias(f"{name}__month"),
                    pl.col(name).dt.day().alias(f"{name}__day"),
                    pl.col(name).dt.weekday().alias(f"{name}__weekday"),
                ]
            ).drop(name)

        for name, spec in state["categorical"].items():
            if name not in output.columns or name == target:
                continue
            values = output[name].cast(pl.String).fill_null(self.config.categorical_missing_value)
            if self.config.add_missing_indicators and spec["missing"]:
                output = output.with_columns(
                    pl.col(name).is_null().cast(pl.UInt8).alias(f"{name}__missing")
                )
            if spec["kind"] == "one_hot":
                category_exprs = [
                    (values == category).cast(pl.UInt8).alias(f"{name}__{category}")
                    for category in spec["categories"]
                ]
                output = output.with_columns(category_exprs).drop(name)
            else:
                frequencies = spec["frequencies"]
                output = output.with_columns(
                    values.replace_strict(frequencies, default=0.0, return_dtype=pl.Float64).alias(
                        f"{name}__frequency"
                    )
                ).drop(name)

        for name, spec in state["scaling"].items():
            if name in output.columns and name != target:
                output = output.with_columns(
                    ((pl.col(name) - spec["center"]) / spec["spread"]).alias(name)
                )
        return output

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted_:
            raise DataDocError("Pipeline is not fitted. Call fit() on training data first.")
        return self._transform_with_state(df, self.state_)

    def fit_transform(self, train_df: pl.DataFrame, target: str | None = None) -> pl.DataFrame:
        return self.fit(train_df, target=target).transform(train_df)

    def evaluate(
        self, df: pl.DataFrame, target: str | None = None, test_size: float = 0.2
    ) -> EvaluationReport:
        df = _normalize_numeric_missing(df)
        target = target or self.config.target
        if not target:
            raise DataDocError("Evaluation requires an explicit target column.")
        if target not in df.columns:
            raise DataDocError(f"Target column '{target}' is not present in evaluation data.")
        if df[target].null_count():
            raise DataDocError(
                "Evaluation target contains missing values. Resolve labels before benchmarking."
            )
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.metrics import balanced_accuracy_score, mean_squared_error
            from sklearn.model_selection import (
                GroupKFold,
                GroupShuffleSplit,
                KFold,
                StratifiedKFold,
                TimeSeriesSplit,
                train_test_split,
            )
        except ImportError as error:
            raise DataDocError(
                "Evaluation requires the optional ML dependencies: pip install 'datadoc-cli[ml]'."
            ) from error

        target_values = df[target]
        inferred_task = self.config.task
        if inferred_task == "auto":
            inferred_task = (
                "regression"
                if target_values.dtype.is_numeric() and target_values.n_unique() > 20
                else "classification"
            )
        if df.height < 10:
            raise DataDocError("Evaluation requires at least 10 rows.")
        indices = list(range(df.height))
        split_strategy = "random_holdout"
        if self.config.time_column:
            if self.config.time_column not in df.columns:
                raise DataDocError(
                    f"Time column '{self.config.time_column}' is not present in evaluation data."
                )
            df = df.sort(self.config.time_column)
            cutoff = max(1, int(df.height * (1 - test_size)))
            train_indices, test_indices = list(range(cutoff)), list(range(cutoff, df.height))
            split_strategy = "ordered_holdout"
        elif self.config.group_column:
            if self.config.group_column not in df.columns:
                raise DataDocError(
                    f"Group column '{self.config.group_column}' is not present in evaluation data."
                )
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.config.random_seed
            )
            train_indices, test_indices = next(
                splitter.split(indices, groups=df[self.config.group_column].to_numpy())
            )
            split_strategy = "group_holdout"
        else:
            stratify = (
                target_values.to_list()
                if inferred_task == "classification" and target_values.n_unique() > 1
                else None
            )
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=self.config.random_seed,
                stratify=stratify,
            )
            if inferred_task == "classification":
                split_strategy = "stratified_holdout"
        train_df, test_df = df[train_indices], df[test_indices]

        baseline_config = PipelineConfig(
            **{**asdict(self.config), "clip_outliers": False, "scaling": "none"}
        )

        evaluation_warnings: set[str] = set()

        def score(
            fit_df: pl.DataFrame, validation_df: pl.DataFrame, config: PipelineConfig
        ) -> float:
            pipeline = DataDocPipeline(config).fit(fit_df, target=target)
            transformed_train = pipeline.transform(fit_df)
            transformed_validation = pipeline.transform(validation_df)
            feature_columns = [
                name
                for name, dtype in transformed_train.schema.items()
                if name != target and dtype.is_numeric()
            ]
            ignored_columns = [
                name
                for name in transformed_train.columns
                if name != target and name not in feature_columns
            ]
            if ignored_columns:
                evaluation_warnings.add(
                    "Non-numeric transformed columns excluded from estimator input: "
                    + ", ".join(ignored_columns)
                )
            if not feature_columns:
                raise DataDocError(
                    "Evaluation produced no numeric feature columns for the estimator."
                )
            train_features = transformed_train.select(feature_columns)
            validation_features = transformed_validation.select(feature_columns)
            fill_values: dict[str, float] = {}
            for name in feature_columns:
                median = train_features[name].median()
                fill_values[name] = float(median) if median is not None else 0.0
            train_features = train_features.with_columns(
                [
                    pl.col(name).fill_nan(value).fill_null(value).alias(name)
                    for name, value in fill_values.items()
                ]
            )
            validation_features = validation_features.with_columns(
                [
                    pl.col(name).fill_nan(value).fill_null(value).alias(name)
                    for name, value in fill_values.items()
                ]
            )
            x_train = train_features.to_numpy()
            y_train = fit_df[target].to_numpy()
            x_test = validation_features.to_numpy()
            y_test = validation_df[target].to_numpy()
            if inferred_task == "classification":
                model = (
                    LogisticRegression(max_iter=1_000)
                    if config.estimator_family == "linear"
                    else RandomForestClassifier(random_state=config.random_seed)
                )
                model.fit(x_train, y_train)
                return float(balanced_accuracy_score(y_test, model.predict(x_test)))
            model = (
                Ridge()
                if config.estimator_family == "linear"
                else RandomForestRegressor(random_state=config.random_seed)
            )
            model.fit(x_train, y_train)
            return -float(mean_squared_error(y_test, model.predict(x_test)) ** 0.5)

        if self.config.time_column:
            folds = TimeSeriesSplit(n_splits=3).split(train_df)
        elif self.config.group_column:
            folds = GroupKFold(n_splits=3).split(
                train_df, groups=train_df[self.config.group_column].to_numpy()
            )
        elif inferred_task == "classification":
            folds = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=self.config.random_seed
            ).split(train_df, train_df[target].to_numpy())
        else:
            folds = KFold(n_splits=3, shuffle=True, random_state=self.config.random_seed).split(
                train_df
            )

        baseline_cv, candidate_cv = [], []
        for fit_indices, validation_indices in folds:
            fold_train, fold_validation = train_df[fit_indices], train_df[validation_indices]
            baseline_cv.append(score(fold_train, fold_validation, baseline_config))
            candidate_cv.append(score(fold_train, fold_validation, self.config))
        selected_config = (
            self.config
            if sum(candidate_cv) / len(candidate_cv) >= sum(baseline_cv) / len(baseline_cv)
            else baseline_config
        )
        baseline_score = score(train_df, test_df, baseline_config)
        selected_score = score(train_df, test_df, selected_config)
        metric = "balanced_accuracy" if inferred_task == "classification" else "negative_rmse"
        return EvaluationReport(
            task=inferred_task,
            metric=metric,
            estimator_family=self.config.estimator_family,
            split_strategy=split_strategy,
            baseline_score=baseline_score,
            selected_score=selected_score,
            improvement=selected_score - baseline_score,
            selected_pipeline="candidate" if selected_config is self.config else "baseline",
            feature_count=len(
                DataDocPipeline(selected_config).fit(train_df, target=target).output_schema_
            )
            - 1,
            warnings=[
                "Holdout score is an estimate; use an external final test set for production decisions."
            ]
            + sorted(evaluation_warnings),
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted_:
            raise DataDocError("Only fitted pipelines can be saved.")
        return {
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "input_schema": self.input_schema_,
            "output_schema": self.output_schema_,
            "state": self.state_,
            "profile": self.profile_.to_dict() if self.profile_ else None,
            "plan": self.plan_.to_dict() if self.plan_ else None,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DataDocPipeline":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise DataDocError(
                "Pipeline artifact version is not supported by this DATADOC version."
            )
        pipeline = cls(PipelineConfig(**artifact["config"]))
        pipeline.input_schema_ = artifact["input_schema"]
        pipeline.output_schema_ = artifact["output_schema"]
        pipeline.state_ = artifact["state"]
        pipeline.fitted_ = True
        return pipeline

    def export_python(self, artifact_path: str | Path) -> str:
        artifact_path = str(artifact_path).replace("\\", "/")
        return f"""import polars as pl
from datadoc.core.pipeline import DataDocPipeline, read_dataset, write_dataset

PIPELINE_PATH = {artifact_path!r}

def transform_file(input_path: str, output_path: str) -> None:
    pipeline = DataDocPipeline.load(PIPELINE_PATH)
    transformed = pipeline.transform(read_dataset(input_path))
    write_dataset(transformed, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply a fitted DATADOC pipeline.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    transform_file(args.input_path, args.output_path)
"""
