"""DATADOC AI package for intelligent feature engineering and dataset explanations."""

from datadoc.ai.client import get_llm_client
from datadoc.ai.explainer import AIExplainer, explain_dataset

__all__ = ["AIExplainer", "explain_dataset", "get_llm_client"]
