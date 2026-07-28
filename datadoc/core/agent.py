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

CRITICAL RULES:
1. DO NOT train any machine learning models (no LinearRegression, Ridge, Lasso, etc.).
2. DO NOT evaluate metrics (no R², MAE).
3. DO NOT create plots or graphs.
4. DO NOT generate fake, synthetic sample datasets. 
Your ONLY job is to apply data cleaning and feature engineering transformations (imputing, encoding, scaling, dropping) to the actual `df` to return a clean dataset so the user can train their model LATER.

INTERVIEW GUIDELINES:
- Start the conversation by briefly introducing yourself and asking what model they are planning to train and what their target variable is.
- Ask a MAXIMUM of 2 questions in a single message.
- Once you know their goal and target variable, STOP asking questions. Immediately state: "I have enough context. Please type `plan` to proceed."
- Be extremely concise, professional, and highly analytical."""
            }
        ]
        
    def chat_step(self, user_input: str) -> str:
        if user_input:
            self.messages.append({"role": "user", "content": user_input})
            
        try:
            response = litellm.completion(
                model=self.model,
                messages=self.messages,
                api_key=self.api_key
            )
            msg = response.choices[0].message.content
        except Exception as e:
            msg = f"❌ Network or API Error: Could not reach the LLM provider. Please check your internet connection or API Key.\n\nDetails: {str(e)}"
            
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
CRITICAL RULES:
- You must apply transformations to the `df` passed to the function.
- DO NOT create a sample dataset using `pd.DataFrame({...})`.
- You can convert to pandas if you prefer (`df = df.to_pandas()`) but the final return should be the dataframe.
- You should orchestrate DATADOC plugins for standard tasks (e.g. `df = ScalingPlugin().apply(df)`), and write custom Pandas/Polars code for bespoke transformations.
Output ONLY the python code inside a ```python ``` block. Do not include extra text."""
        
        response_msg = self.chat_step(prompt)
        return self._extract_code(response_msg)
        
    def evaluate_and_fix(self, error_traceback: str) -> tuple[str, str]:
        """Returns (reasoning, new_code)"""
        prompt = f"""The generated code failed to execute with the following error:
```
{error_traceback}
```
Please reason about WHY this error occurred.
First, explain your thought process and how you intend to fix it.
Then, provide the completely rewritten, corrected `clean_data(df)` function inside a ```python ``` block.

DO NOT generate fake data or train ML models. Just fix the data transformation logic.
"""
        response_msg = self.chat_step(prompt)
        
        # Split reasoning and code
        code = self._extract_code(response_msg)
        # Remove the code block from the reasoning to just get the text
        reasoning = re.sub(r"```(?:python)?\s*.*?```", "", response_msg, flags=re.DOTALL).strip()
        
        return reasoning, code
        
    def _extract_code(self, response_msg: str) -> str:
        code_block = re.search(r"```(?:python)?\s*(.*?)```", response_msg, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        return response_msg.strip()
