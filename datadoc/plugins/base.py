from abc import ABC, abstractmethod
import polars as pl

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


    @abstractmethod
    def analyze(self, df: pl.DataFrame) -> dict:
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
    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the engineering transformation and return the new dataframe."""
        pass



    def explain(self) -> str:
        """Return a human-readable explanation of what this plugin does."""
        return f"{self.name} (v{self.version}): {self.description}"

