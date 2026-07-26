import pandas as pd\nimport numpy as np\n\ndef load_and_clean_data(file_path: str = 'test.csv') -> pd.DataFrame:\n    df = pd.read_csv(file_path)\n\n    # Missing Value Imputation
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])\n\n    # Categorical Encoding
cat_cols = ['name', 'department']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)\n\n    return df\n\nif __name__ == '__main__':\n    clean_df = load_and_clean_data()\n    print(f'Successfully processed {len(clean_df)} rows!')