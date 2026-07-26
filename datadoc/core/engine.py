import pandas as pd
from typing import Dict, Any
from datadoc.plugins.missing_values import MissingValuePlugin
from datadoc.plugins.encoders import CategoricalEncoderPlugin
from datadoc.plugins.outliers import OutlierPlugin

class DATADOC:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        
        # Hardcoded rule engine for MVP
        self.plugins = [
            MissingValuePlugin(),
            OutlierPlugin(),
            CategoricalEncoderPlugin()
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
        
    def pipeline(self) -> str:
        """Generates the full Python script representing the pipeline."""
        script = [
            "import pandas as pd",
            "import numpy as np",
            "",
            f"def load_and_clean_data(file_path: str = '{self.file_path}') -> pd.DataFrame:",
            "    df = pd.read_csv(file_path)",
            ""
        ]
        
        for plugin in self.plugins:
            analysis = plugin.analyze(self.df)
            should_apply = any(v for k, v in analysis.items() if k.startswith('has_') and v is True)
            if should_apply:
                code_snippet = plugin.generate_code(analysis)
                if code_snippet:
                    # Indent code snippet properly
                    indented = "\\n".join([f"    {line}" for line in code_snippet.split("\\n") if line])
                    script.append(indented)
                    script.append("")
                    
        script.append("    return df")
        script.append("")
        script.append("if __name__ == '__main__':")
        script.append("    clean_df = load_and_clean_data()")
        script.append("    print(f'Successfully processed {len(clean_df)} rows!')")
        
        return "\\n".join(script)
        
    def engineer(self) -> pd.DataFrame:
        """Automatically triggers plugins that are needed."""
        df_transformed = self.df.copy()
        
        for plugin in self.plugins:
            analysis = plugin.analyze(df_transformed)
            # Generic rule trigger: if any 'has_' flag is True
            should_apply = any(v for k, v in analysis.items() if k.startswith('has_') and v is True)
            if should_apply:
                df_transformed = plugin.apply(df_transformed)
                
        return df_transformed
