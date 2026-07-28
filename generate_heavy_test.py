import polars as pl
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
import sys
import io

# Fix unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datadoc.core.engine import DATADOC

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

N = 10000

print(f"Generating heavy test dataset with {N} rows...")

# 1. ID column
user_id = np.arange(1, N + 1)

# 2. String ID/UUID column
transaction_id = [str(uuid.uuid4()) for _ in range(N)]

# 3. High cardinality string (Name)
names = [f"User_{i}_{random.randint(1000, 9999)}" for i in range(N)]

# 4. Age (numeric, some missing, some outliers)
age = np.random.normal(35, 10, N)
age[np.random.choice(N, 500, replace=False)] = np.nan # 5% missing
age[np.random.choice(N, 10, replace=False)] = 999 # Outliers

# 5. Salary (numeric, skewed, some missing, massive outliers)
salary = np.random.lognormal(mean=11, sigma=0.5, size=N)
salary[np.random.choice(N, 800, replace=False)] = np.nan # 8% missing
salary[np.random.choice(N, 5, replace=False)] = 1e9 # Outliers

# 6. Category (low cardinality string, some missing)
categories = ['Electronics', 'Clothing', 'Home', 'Toys']
category = np.random.choice(categories, N, p=[0.4, 0.3, 0.2, 0.1])
category = np.where(np.random.rand(N) < 0.05, None, category)

# 7. is_active (binary 0/1)
is_active = np.random.choice([0, 1], N, p=[0.2, 0.8])
is_active = np.where(np.random.rand(N) < 0.02, None, is_active) # some missing

# 8. Signup Date (datetime string)
base_date = datetime(2023, 1, 1)
signup_date = [base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23)) for _ in range(N)]
signup_date = [d.strftime("%Y-%m-%d %H:%M:%S") if random.random() > 0.05 else None for d in signup_date]

# 9. Constant Column
constant_col = ["VERSION_1"] * N

# Build DataFrame
df = pl.DataFrame({
    "user_id": user_id,
    "transaction_id": transaction_id,
    "first_name": names,
    "age": age,
    "salary": salary,
    "category": category.tolist(),
    "is_active": is_active.astype(float),
    "signup_date": signup_date,
    "constant_col": constant_col
})

df.write_csv("heavy_test.csv")
print("Saved to heavy_test.csv")
print("Original Shape:", df.shape)
print("Original Columns:", df.columns)
print()

print("Running DATADOC Pipeline...")
doc = DATADOC("heavy_test.csv")

def on_progress(name, status, details):
    if status == "applied":
        print(f"[{name}] {details[0]}")
    elif status == "skipped":
        print(f"[{name}] Skipped")

clean_df = doc.engineer(progress_callback=on_progress)

print("\n=== FINAL CLEANED DATASET ===")
print("Shape:", clean_df.shape)
print("Columns:", clean_df.columns)
print("Null count per column:")
nulls = {col: clean_df[col].null_count() for col in clean_df.columns}
for k, v in nulls.items():
    print(f"  {k}: {v}")

print("\nSample Data:")
print(clean_df.head())

# Checks for ML readiness
print("\n=== ML READINESS CHECKS ===")
errors = []
for col in clean_df.columns:
    if clean_df[col].dtype == pl.String:
        errors.append(f"String column remains: {col}")
    if clean_df[col].null_count() > 0:
        errors.append(f"Nulls remain in: {col}")

if errors:
    print("WARNING: Dataset may not be perfectly ML-ready:")
    for e in errors:
        print(f" - {e}")
else:
    print("SUCCESS: Dataset is strictly numeric with no missing values (ML-Ready).")
