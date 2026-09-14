"""Leakage-safe, serializable preparation pipelines for tabular data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal
import warnings

import polars as pl


ARTIFACT_VERSION = 2
ID_NAME_PATTERN = re.compile(
    r"(?:(?:^|[_\-\s])(id|index|key|uuid)$|(?<=[a-z])(Id|ID)$)", re.IGNORECASE
)
DATETIME_NAME_PATTERN = re.compile(r"(?:date|time|timestamp|_at)$", re.IGNORECASE)
RARE_TOKEN = "__RARE__"

try:
    from datadoc import __version__ as _DATADOC_VERSION  # type: ignore
except Exception:
    _DATADOC_VERSION = None


def _datadoc_version() -> str:
    if _DATADOC_VERSION:
        return str(_DATADOC_VERSION)
    try:
        from pathlib import Path as _P

        for line in (
            (_P(__file__).resolve().parents[2] / "pyproject.toml")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            if line.strip().startswith("version ="):
                return line.split('"', 2)[1]
    except Exception:
        pass
    return "unknown"


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
    deduplicate: bool = False
    categorical_threshold: int = 20
    datetime_parse_threshold: float = 0.9
    datetime_extract_hour: bool = True
    datetime_cyclical: bool = False
    add_missing_indicators: bool = True
    categorical_missing_value: str = "__MISSING__"
    rare_category_min_frequency: float = 0.0
    rare_token: str = "__RARE__"
    encode_high_cardinality: bool = True
    clip_outliers: bool = False
    outlier_multiplier: float = 1.5
    scaling: Literal["none", "standard", "robust", "auto"] = "auto"
    estimator_family: Literal["linear", "tree", "both"] = "linear"
    random_seed: int = 42
    strict_schema: bool = True
    time_column: str | None = None
    group_column: str | None = None
    custom_interactions: list[dict[str, str]] = field(default_factory=list)
    custom_transforms: list[dict[str, str]] = field(default_factory=list)

    def resolved_scaling(self) -> Literal["none", "standard", "robust"]:
        if self.scaling != "auto":
            return self.scaling
        # "both" evaluates a linear model too, so standard scaling is the safe default
        # (trees are scale-invariant, linear models are not).
        return "standard" if self.estimator_family in ("linear", "both") else "none"


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
    if not path.exists():
        raise DataDocError(f"Dataset not found: '{path}'. Please check the file path.")
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


def _dataset_provenance(df: pl.DataFrame, schema_fp: str | None = None) -> dict[str, Any]:
    """Lightweight train provenance: fingerprint + rows + cols + column hash."""
    fp = schema_fp or _schema_fingerprint(df)
    col_hash = hashlib.sha256(",".join(df.columns).encode("utf-8")).hexdigest()[:16]
    return {
        "schema_fingerprint": fp,
        "columns_hash": col_hash,
        "rows": df.height,
        "columns": df.width,
        "column_names": list(df.columns),
    }


def _has_time_component_series(series: pl.Series) -> bool:
    """True if a Date/Datetime series has non-midnight time info."""
    try:
        times = series.drop_nulls()
        if times.len() == 0:
            return False
        if times.dtype == pl.Date:
            return False
        hours = times.dt.hour()
        minutes = times.dt.minute()
        return not ((hours == 0).all() and (minutes == 0).all())
    except Exception:
        return False


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
        # High null rate finding (>50%)
        if df.height and null_count / df.height > 0.5:
            findings.append(
                {
                    "severity": "warning",
                    "column": name,
                    "code": "high_null_rate",
                    "message": f"Column has {null_count / df.height:.0%} missing values.",
                }
            )
        # Identifier detection: explicit list or name-pattern (including camelCase like PassengerId)
        if name in config.identifier_columns or ID_NAME_PATTERN.search(name):
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
                # High-cardinality warning for categoricals
                if unique > config.categorical_threshold:
                    findings.append(
                        {
                            "severity": "info",
                            "column": name,
                            "code": "high_cardinality",
                            "message": f"Column has {unique} distinct values (threshold={config.categorical_threshold}); will be frequency-encoded.",
                        }
                    )
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
        self.train_provenance_: dict[str, Any] = {}
        self.fitted_ = False

    def profile(self, df: pl.DataFrame) -> DatasetProfile:
        self.profile_ = profile_dataset(df, self.config)
        return self.profile_

    def plan(self, df: pl.DataFrame) -> TransformPlan:
        profile = self.profile(df)
        operations: list[dict[str, Any]] = []
        # deduplication is dataset-level, surface first
        if profile.duplicate_rows and self.config.deduplicate:
            operations.append(
                {
                    "operation": "deduplicate",
                    "column": "*",
                    "reason": f"Drop {profile.duplicate_rows} duplicate rows at fit time.",
                }
            )
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
                detail = "Median learned from training data."
                if self.config.clip_outliers:
                    detail += f" IQR clipping (x{self.config.outlier_multiplier}) enabled."
                operations.append(
                    {
                        "operation": "numeric_imputation",
                        "column": role.name,
                        "reason": detail,
                    }
                )
                if self.config.clip_outliers:
                    operations.append(
                        {
                            "operation": "outlier_clipping",
                            "column": role.name,
                            "reason": f"Clip to Q1-{self.config.outlier_multiplier}*IQR / Q3+{self.config.outlier_multiplier}*IQR (train-only).",
                        }
                    )
            elif role.role == "feature_categorical":
                detail = "Vocabulary learned from training data."
                if self.config.rare_category_min_frequency > 0:
                    detail += f" Rare < {self.config.rare_category_min_frequency:.1%} grouped into {self.config.rare_token}."
                operations.append(
                    {
                        "operation": "categorical_encoding",
                        "column": role.name,
                        "reason": detail,
                    }
                )
            elif role.role == "feature_datetime":
                feats = "year/month/day/weekday"
                if self.config.datetime_extract_hour:
                    feats += " (+hour when time present)"
                if self.config.datetime_cyclical:
                    feats += " + cyclical sin/cos"
                operations.append(
                    {
                        "operation": "datetime_features",
                        "column": role.name,
                        "reason": f"Calendar features ({feats}) from parsed timestamps.",
                    }
                )
        scaling = self.config.resolved_scaling()
        if scaling != "none":
            operations.append(
                {
                    "operation": f"{scaling}_scaling",
                    "column": "*",
                    "reason": f"{scaling.title()} scaling (center/spread from train only, binaries excluded).",
                }
            )
        for interaction in self.config.custom_interactions:
            col_a = interaction.get("col_a")
            col_b = interaction.get("col_b")
            op = interaction.get("op")
            rationale = interaction.get("rationale") or f"Discovered interaction: {col_a} {op} {col_b}"
            operations.append(
                {
                    "operation": "custom_interaction",
                    "column": f"{col_a}_{op}_{col_b}",
                    "reason": rationale,
                }
            )

        for transform in self.config.custom_transforms:
            col = transform.get("col")
            t_name = transform.get("transform")
            rationale = transform.get("rationale") or f"Discovered transform: {t_name}({col})"
            operations.append(
                {
                    "operation": "custom_transform",
                    "column": f"{col}_{t_name}",
                    "reason": rationale,
                }
            )

        self.plan_ = TransformPlan(operations, profile.findings, self.config.protected_columns)
        return self.plan_

    def explain_plan(self, df: pl.DataFrame) -> str:
        """Human-readable English trace of the plan (for --explain / notebooks)."""
        plan = self.plan(df)
        lines = [f"Plan for {df.height} rows x {df.width} cols (target={self.config.target}):"]
        for i, op in enumerate(plan.operations, 1):
            lines.append(f"  {i:02d}. {op['operation']:22s} {op['column']:24s} — {op['reason']}")
        if plan.findings:
            lines.append(f"Findings ({len(plan.findings)}):")
            for f in plan.findings[:10]:
                lines.append(f"  - [{f['severity']}] {f['code']} ({f['column']}): {f['message']}")
        return "\n".join(lines)

    def fit(self, train_df: pl.DataFrame, target: str | None = None) -> "DataDocPipeline":
        train_df = _normalize_numeric_missing(train_df)
        if target is not None:
            self.config.target = target
        if self.config.target and self.config.target not in train_df.columns:
            raise DataDocError(
                f"Target column '{self.config.target}' is not present in training data."
            )
        # Deduplicate train-only (never touch validation at transform)
        _deduped = 0
        if self.config.deduplicate and train_df.height:
            before = train_df.height
            train_df = train_df.unique(maintain_order=True)
            _deduped = before - train_df.height

        profile = self.profile(train_df)
        self.plan(train_df)
        # plan() runs on post-dedup data, so re-inject the op when rows were dropped
        if _deduped and not any(
            op.get("operation") == "deduplicate" for op in self.plan_.operations
        ):
            self.plan_.operations.insert(
                0,
                {
                    "operation": "deduplicate",
                    "column": "*",
                    "reason": f"Dropped {_deduped} duplicate rows at fit time.",
                },
            )
        self.input_schema_ = {name: str(dtype) for name, dtype in train_df.schema.items()}
        self.train_provenance_ = _dataset_provenance(train_df, profile.schema_fingerprint)
        self.train_provenance_["deduplicated_rows"] = _deduped
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
            if (
                role.role == "constant"
                or role.role == "text"
                or (role.role == "identifier" and self.config.drop_identifiers)
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
                rel = {k: c / total for k, c in frequencies.items()}
                rare_cut = self.config.rare_category_min_frequency
                # Group rare into token so one-hot width stays capped but signal kept
                rare_values = (
                    sorted([k for k, f in rel.items() if f < rare_cut]) if rare_cut > 0 else []
                )
                kept = (
                    sorted([k for k, f in rel.items() if f >= rare_cut])
                    if rare_cut > 0
                    else sorted(rel.keys())
                )
                if rare_values:
                    # ensure token present as explicit category
                    if self.config.rare_token not in kept:
                        kept = sorted(kept + [self.config.rare_token])
                if len(kept) <= self.config.categorical_threshold:
                    state["categorical"][name] = {
                        "kind": "one_hot",
                        "categories": kept,
                        "rare_values": rare_values,
                        "rare_token": self.config.rare_token,
                        "missing": series.null_count() > 0,
                    }
                elif self.config.encode_high_cardinality:
                    state["categorical"][name] = {
                        "kind": "frequency",
                        "frequencies": rel,
                        "rare_values": rare_values,
                        "rare_token": self.config.rare_token,
                        "missing": series.null_count() > 0,
                    }
                else:
                    state["dropped"].append(name)
            elif role.role == "feature_datetime":
                # Detect hour presence on train (parsed if needed)
                has_hour = False
                try:
                    s = series
                    if s.dtype == pl.String:
                        s = s.cast(pl.String).str.to_datetime(strict=False)
                    if s.dtype in (pl.Date, pl.Datetime):
                        has_hour = _has_time_component_series(s)
                except Exception:
                    has_hour = False
                state["datetime"][name] = {
                    "source_dtype": str(series.dtype),
                    "has_hour": bool(has_hour and self.config.datetime_extract_hour),
                    "cyclical": bool(self.config.datetime_cyclical),
                }

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
            # Build inf→null expression first so missing indicator catches ±inf
            inf_expr = pl.col(name)
            if output[name].dtype.is_float():
                inf_expr = pl.when(inf_expr.is_infinite()).then(None).otherwise(inf_expr)
                output = output.with_columns(inf_expr.alias(name))
            if self.config.add_missing_indicators and spec["missing"]:
                output = output.with_columns(
                    pl.col(name).is_null().cast(pl.UInt8).alias(f"{name}__missing")
                )
            expr = pl.col(name).fill_null(spec["median"])
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
            feats = [
                pl.col(name).dt.year().alias(f"{name}__year"),
                pl.col(name).dt.month().alias(f"{name}__month"),
                pl.col(name).dt.day().alias(f"{name}__day"),
                pl.col(name).dt.weekday().alias(f"{name}__weekday"),
            ]
            # Hour only when train had real time info (backwards compat: default True for v1 artifacts)
            if spec.get("has_hour", True):
                try:
                    feats.append(pl.col(name).dt.hour().alias(f"{name}__hour"))
                except Exception:
                    pass
            output = output.with_columns(feats).drop(name)
            # Optional cyclical encodings (month/day/weekday/hour -> sin/cos)
            if spec.get("cyclical", False):
                import math as _math

                cyc_exprs = []
                periods = {
                    f"{name}__month": 12.0,
                    f"{name}__day": 31.0,
                    f"{name}__weekday": 7.0,
                }
                if f"{name}__hour" in output.columns:
                    periods[f"{name}__hour"] = 24.0
                for col, period in periods.items():
                    if col in output.columns:
                        cyc_exprs.append(
                            (2.0 * _math.pi * pl.col(col) / period).sin().alias(f"{col}__sin")
                        )
                        cyc_exprs.append(
                            (2.0 * _math.pi * pl.col(col) / period).cos().alias(f"{col}__cos")
                        )
                if cyc_exprs:
                    output = output.with_columns(cyc_exprs)

        for name, spec in state["categorical"].items():
            if name not in output.columns or name == target:
                continue
            values = output[name].cast(pl.String).fill_null(self.config.categorical_missing_value)
            # Map rare/unseen to token: rare_values learned on train; unseen -> rare token if configured
            rare_values = spec.get("rare_values", [])
            rare_token = spec.get("rare_token", self.config.rare_token)
            if rare_values:
                rare_set = set(rare_values)
                values = (
                    pl.when(values.is_in(list(rare_set))).then(pl.lit(rare_token)).otherwise(values)
                )
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
                # Unseen categories -> 0.0; rare token uses its train frequency if present
                output = output.with_columns(
                    values.replace_strict(frequencies, default=0.0, return_dtype=pl.Float64).alias(
                        f"{name}__frequency"
                    )
                ).drop(name)

        # Apply custom interactions (Agent proposed)
        for interaction in self.config.custom_interactions:
            col_a = interaction.get("col_a")
            col_b = interaction.get("col_b")
            op = interaction.get("op")
            if col_a in output.columns and col_b in output.columns:
                try:
                    # ensure numeric
                    expr_a = pl.col(col_a).cast(pl.Float64)
                    expr_b = pl.col(col_b).cast(pl.Float64)
                    if op == "add":
                        expr = expr_a + expr_b
                    elif op == "sub":
                        expr = expr_a - expr_b
                    elif op == "mul":
                        expr = expr_a * expr_b
                    elif op == "div":
                        # avoid div by zero
                        expr = expr_a / (expr_b.replace(0.0, None))
                    else:
                        continue
                    output = output.with_columns(expr.alias(f"{col_a}_{op}_{col_b}"))
                except Exception:
                    pass

        # Apply custom transforms (Agent proposed)
        for transform in self.config.custom_transforms:
            col = transform.get("col")
            op = transform.get("transform")
            if col in output.columns:
                try:
                    expr = pl.col(col).cast(pl.Float64)
                    if op == "log1p":
                        # avoid log of negative
                        expr = pl.when(expr >= -1).then((expr + 1).log()).otherwise(None)
                    elif op == "sqrt":
                        expr = pl.when(expr >= 0).then(expr.sqrt()).otherwise(None)
                    elif op == "square":
                        expr = expr * expr
                    else:
                        continue
                    output = output.with_columns(expr.alias(f"{col}_{op}"))
                except Exception:
                    pass

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                family = config.estimator_family
                if inferred_task == "classification":
                    if family == "both":
                        # Honest "both": fit linear and tree, keep the stronger validation score.
                        scores = []
                        for model in (
                            LogisticRegression(max_iter=1_000),
                            RandomForestClassifier(random_state=config.random_seed),
                        ):
                            model.fit(x_train, y_train)
                            scores.append(
                                float(balanced_accuracy_score(y_test, model.predict(x_test)))
                            )
                        return max(scores)
                    model = (
                        LogisticRegression(max_iter=1_000)
                        if family == "linear"
                        else RandomForestClassifier(random_state=config.random_seed)
                    )
                    model.fit(x_train, y_train)
                    return float(balanced_accuracy_score(y_test, model.predict(x_test)))
                if family == "both":
                    scores = []
                    for model in (
                        Ridge(),
                        RandomForestRegressor(random_state=config.random_seed),
                    ):
                        model.fit(x_train, y_train)
                        scores.append(
                            -float(mean_squared_error(y_test, model.predict(x_test)) ** 0.5)
                        )
                    return max(scores)
                model = (
                    Ridge()
                    if family == "linear"
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
        provenance = getattr(self, "train_provenance_", None)
        if provenance is None and self.profile_ is not None:
            provenance = _dataset_provenance(
                pl.DataFrame(schema={k: pl.String for k in self.input_schema_}),
                self.profile_.schema_fingerprint,
            )
            provenance["rows"] = self.profile_.rows
        return {
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "datadoc_version": _datadoc_version(),
            "config": asdict(self.config),
            "input_schema": self.input_schema_,
            "output_schema": self.output_schema_,
            "state": self.state_,
            "provenance": provenance or {},
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
        version = artifact.get("artifact_version")
        if version not in (1, 2, ARTIFACT_VERSION):
            raise DataDocError(
                f"Pipeline artifact version {version} is not supported by this DATADOC version."
            )
        raw_cfg = dict(artifact["config"])
        # Backwards compat: v1 artifacts lack new fields -> fill defaults
        cfg = PipelineConfig(
            **{k: v for k, v in raw_cfg.items() if k in PipelineConfig.__dataclass_fields__}
        )
        pipeline = cls(cfg)
        pipeline.input_schema_ = artifact["input_schema"]
        pipeline.output_schema_ = artifact["output_schema"]
        pipeline.state_ = artifact["state"]
        # Backfill datetime spec defaults for v1 artifacts
        for _spec in pipeline.state_.get("datetime", {}).values():
            _spec.setdefault("has_hour", True)
            _spec.setdefault("cyclical", False)
        for _spec in pipeline.state_.get("categorical", {}).values():
            _spec.setdefault("rare_values", [])
            _spec.setdefault("rare_token", cfg.rare_token)
        pipeline.train_provenance_ = artifact.get("provenance", {})
        pipeline.fitted_ = True
        return pipeline

    # ── Addictive-loop helpers ──────────────────────────────
    def drift_report(self, df: pl.DataFrame) -> dict[str, Any]:
        """Schema + median-shift drift vs train provenance (for --validate)."""
        if not self.fitted_:
            raise DataDocError("Pipeline is not fitted.")
        df = _normalize_numeric_missing(df)
        issues: list[dict[str, Any]] = []
        try:
            self._validate_input_schema(df)
            schema_ok = True
        except DataDocError as e:
            schema_ok = False
            issues.append({"type": "schema", "message": str(e)})
        for name, spec in self.state_.get("numeric", {}).items():
            if name not in df.columns or not df[name].dtype.is_numeric():
                continue
            median = spec.get("median")
            try:
                cur = df[name].median()
            except Exception:
                cur = None
            if median is not None and cur is not None:
                drift = abs(float(cur) - float(median)) / (abs(float(median)) + 1e-9)
                if drift > 0.5:
                    issues.append(
                        {
                            "type": "drift",
                            "column": name,
                            "train_median": median,
                            "current_median": float(cur),
                            "shift": drift,
                            "message": f"{name} median shift {drift:.1%} ({median} -> {cur})",
                        }
                    )
        return {
            "schema_ok": schema_ok,
            "issues": issues,
            "provenance": getattr(self, "train_provenance_", {}),
        }

    def evaluate_ablation(self, df: pl.DataFrame, target: str | None = None) -> dict[str, Any]:
        """Per-component ablation: baseline vs no-clip vs no-scale vs full candidate."""
        base_report = self.evaluate(df, target=target).to_dict()
        variants: dict[str, Any] = {"full": base_report}
        try:
            no_clip_cfg = PipelineConfig(**{**asdict(self.config), "clip_outliers": False})
            variants["no_clip"] = (
                DataDocPipeline(no_clip_cfg)
                .evaluate(df, target=target or self.config.target)
                .to_dict()
            )
        except Exception as e:
            variants["no_clip"] = {"error": str(e)}
        try:
            no_scale_cfg = PipelineConfig(**{**asdict(self.config), "scaling": "none"})
            variants["no_scaling"] = (
                DataDocPipeline(no_scale_cfg)
                .evaluate(df, target=target or self.config.target)
                .to_dict()
            )
        except Exception as e:
            variants["no_scaling"] = {"error": str(e)}
        try:
            minimal_cfg = PipelineConfig(
                **{
                    **asdict(self.config),
                    "scaling": "none",
                    "clip_outliers": False,
                    "rare_category_min_frequency": 0.0,
                }
            )
            variants["minimal"] = (
                DataDocPipeline(minimal_cfg)
                .evaluate(df, target=target or self.config.target)
                .to_dict()
            )
        except Exception as e:
            variants["minimal"] = {"error": str(e)}
        return {
            "metric": base_report.get("metric"),
            "variants": {
                k: (
                    {
                        "baseline_score": v.get("baseline_score"),
                        "selected_score": v.get("selected_score"),
                        "improvement": v.get("improvement"),
                        "selected_pipeline": v.get("selected_pipeline"),
                    }
                    if isinstance(v, dict) and "error" not in v
                    else v
                )
                for k, v in variants.items()
            },
        }

    def profile_to_html(self, df: pl.DataFrame | None = None) -> str:
        """Small HTML widget for notebooks (Polars-native, no heavy deps)."""
        prof = self.profile_ or (self.profile(df) if df is not None else None)
        if prof is None:
            raise DataDocError("No profile available. Call profile(df) first.")
        rows = "".join(
            f"<tr><td><b>{r.name}</b></td><td>{r.role}</td><td>{r.confidence:.2f}</td><td>{r.rationale}</td></tr>"
            for r in prof.roles
        )
        findings = (
            "".join(
                f"<li><b>[{f['severity']}] {f['code']}</b> ({f['column']}): {f['message']}</li>"
                for f in prof.findings
            )
            or "<li>No findings</li>"
        )
        return (
            f"<div><h3>DATADOC Profile — {prof.rows} rows x {prof.columns} cols</h3>"
            f"<p><code>{prof.schema_fingerprint}</code></p>"
            f"<table border='1' cellpadding='4'><tr><th>Column</th><th>Role</th><th>Conf</th><th>Rationale</th></tr>{rows}</table>"
            f"<h4>Findings</h4><ul>{findings}</ul></div>"
        )

    def _repr_html_(self) -> str:
        try:
            if self.profile_ is not None:
                return self.profile_to_html()
        except Exception:
            pass
        return (
            f"<div><b>DataDocPipeline</b> fitted={self.fitted_} target={self.config.target}</div>"
        )

    def export_sklearn_artifact(self, path: str | Path) -> Path:
        """Save a joblib artifact (dict) portable without DATADOC at inference.

        Loads via: joblib.load(path) -> dict with config/state/schemas.
        A tiny loader snippet is returned for docs.
        """
        try:
            import joblib  # type: ignore
        except ImportError as e:
            raise DataDocError("pip install 'datadoc-cli[ml]' for joblib export.") from e
        if not self.fitted_:
            raise DataDocError("Only fitted pipelines can be exported.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_dict(), path)
        return path

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
