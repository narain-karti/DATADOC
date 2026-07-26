import pandas as pd
from typing import Dict, Any
from datadoc.plugins.missing_values import MissingValuePlugin

class DATADOC:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        
        # Hardcoded rule engine for MVP
        self.plugins = [
            MissingValuePlugin()
        ]
        
    def analyze(self) -> Dict[str, Any]:
        """Runs the analyze step across all plugins."""
        report = {
            "rows": self.df.shape[0],
            "cols": self.df.shape[1],
            "plugins": {}
        }
        
        for plugin in self.plugins:
            report["plugins"][plugin.name] = plugin.analyze(self.df)
            
        return report
        
    def recommend(self) -> list[str]:
        """Runs analysis and aggregates recommendations from all plugins."""
        recommendations = []
        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            recs = plugin.recommend(analysis)
            recommendations.extend(recs)
        return recommendations
        
    def engineer(self) -> pd.DataFrame:
        """Automatically triggers plugins that are needed."""
        df_transformed = self.df.copy()
        
        for plugin in self.plugins:
            analysis = plugin.analyze(df_transformed)
            # Simplistic rule trigger
            if analysis.get("has_missing_values"):
                df_transformed = plugin.apply(df_transformed)
                
        return df_transformed
