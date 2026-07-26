from abc import ABC, abstractmethod
import pandas as pd

class BasePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "No description provided."

    @property
    def priority(self) -> int:
        """Lower number = runs first. Default is 50."""
        return 50

    @property
    def supported_datatypes(self) -> list:
        return ["numeric", "categorical"]

    @property
    def dependencies(self) -> list:
        """List of plugin names that must run before this one."""
        return []

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze the dataframe and return metadata/flags if the plugin should run."""
        pass

    @abstractmethod
    def recommend(self, analysis_result: dict) -> list[str]:
        """Return a list of recommendations based on the analysis."""
        pass

    @abstractmethod
    def generate_code(self, analysis_result: dict) -> str:
        """Return the Python code string to replicate this plugin's transformation."""
        pass

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the engineering transformation and return the new dataframe."""
        pass

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that the transformation produced a valid result."""
        if df.empty:
            return False
        return True

    def rollback(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """Return the original dataframe, undoing any transformation."""
        return original_df.copy()

    def explain(self) -> str:
        """Return a human-readable explanation of what this plugin does."""
        return f"{self.name} (v{self.version}): {self.description}"

    def estimate_runtime(self, df: pd.DataFrame) -> float:
        """Estimate runtime in seconds based on dataset size."""
        rows = df.shape[0]
        # Simple linear estimate: ~1ms per 1000 rows
        return round(rows / 1_000_000, 4)
