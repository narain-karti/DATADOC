import litellm
import json
import re
import traceback

class AgenticEngineer:
    def __init__(self, metadata: str, api_key: str, model: str):
        self.metadata = metadata
        self.api_key = api_key
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": f"""You are DATADOC's Principal AI Data Engineer. You are interviewing the user to deeply understand their ML goals before writing a custom dataset engineering pipeline.
Dataset Metadata:
{self.metadata}

Guidelines:
- Start the conversation by briefly introducing yourself and asking what model they are training and what their target variable is.
- Ask exactly ONE question at a time.
- Be extremely concise, professional, and highly analytical."""
            }
        ]
        
    def chat_step(self, user_input: str) -> str:
        if user_input:
            self.messages.append({"role": "user", "content": user_input})
            
        response = litellm.completion(
            model=self.model,
            messages=self.messages,
            api_key=self.api_key
        )
        msg = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": msg})
        return msg
        
    def generate_plan(self) -> str:
        prompt = "Thank you. Based on our conversation, please generate a bulleted 'Implementation Plan' outlining exactly what data cleaning and engineering transformations you will perform on this dataset. Be specific about which columns you will alter. Do not write code yet, just the plan."
        return self.chat_step(prompt)
        
    def generate_code(self) -> str:
        prompt = """Now, write the complete, production-ready Python code to execute this plan.
You MUST write a function with this exact signature:
```python
import polars as pl
import pandas as pd
from datadoc.plugins.missing_values import MissingValuePlugin
from datadoc.plugins.outliers import OutlierPlugin
from datadoc.plugins.datetime_feat import DatetimePlugin
from datadoc.plugins.encoders import CategoricalEncoderPlugin
from datadoc.plugins.scaling import ScalingPlugin

def clean_data(df):
    # Your code here...
    return df
```
You can convert to pandas if you prefer (`df = df.to_pandas()`) but the final return should be the dataframe.
You should orchestrate DATADOC plugins for standard tasks (e.g. `df = ScalingPlugin().apply(df)`), and write custom code for any bespoke, unique transformations needed.
Output ONLY the python code inside a ```python ``` block. Do not include extra text."""
        
        response_msg = self.chat_step(prompt)
        
        # Extract the code from the markdown block
        code_block = re.search(r"```(?:python)?\s*(.*?)```", response_msg, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        return response_msg.strip()
