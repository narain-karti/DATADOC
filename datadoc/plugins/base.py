from abc import ABC, abstractmethod
import pandas as pd

class BasePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze the dataframe and return metadata/flags if the plugin should run."""
        pass

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the engineering transformation and return the new dataframe."""
        pass
