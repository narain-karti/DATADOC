from typing import Literal, Union
from pydantic import BaseModel, Field


class ToggleFeature(BaseModel):
    """Toggle a global pipeline setting on or off."""

    type: Literal["ToggleFeature"]
    setting: Literal[
        "clip_outliers",
        "drop_identifiers",
        "deduplicate",
        "datetime_extract_hour",
        "datetime_cyclical",
        "add_missing_indicators",
        "encode_high_cardinality",
    ] = Field(..., description="The pipeline setting to toggle.")
    value: bool = Field(..., description="The new boolean value for the setting.")
    rationale: str = Field("", description="ML reasoning / hypothesis for this action.")


class SetScaling(BaseModel):
    """Change the numerical scaling strategy."""

    type: Literal["SetScaling"]
    scaling: Literal["none", "standard", "robust", "auto"] = Field(
        ..., description="The scaling strategy to use for numeric features."
    )
    rationale: str = Field("", description="ML reasoning / hypothesis for this action.")


class AddInteraction(BaseModel):
    """Add a mathematical interaction between two numeric columns."""

    type: Literal["AddInteraction"]
    col_a: str = Field(..., description="The name of the first column.")
    col_b: str = Field(..., description="The name of the second column.")
    op: Literal["add", "sub", "mul", "div"] = Field(
        ..., description="The mathematical operation to apply."
    )
    rationale: str = Field("", description="ML reasoning / hypothesis for this action.")


class ApplyTransform(BaseModel):
    """Apply a mathematical transformation to a numeric column."""

    type: Literal["ApplyTransform"]
    col: str = Field(..., description="The name of the column to transform.")
    transform: Literal["log1p", "sqrt", "square"] = Field(
        ..., description="The transformation to apply."
    )
    rationale: str = Field("", description="ML reasoning / hypothesis for this action.")


class IgnoreColumn(BaseModel):
    """Ignore a column entirely during training."""

    type: Literal["IgnoreColumn"]
    col: str = Field(..., description="The name of the column to ignore.")
    rationale: str = Field("", description="ML reasoning / hypothesis for this action.")


class FeatureBatch(BaseModel):
    """A batch of feature engineering hypotheses to test."""

    actions: list[
        Union[ToggleFeature, SetScaling, AddInteraction, ApplyTransform, IgnoreColumn]
    ] = Field(
        ..., description="A list of feature engineering actions to apply to the PipelineConfig."
    )
