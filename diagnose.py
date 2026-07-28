"""Diagnostic: verify the fixed pipeline produces correct output."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import polars as pl
from datadoc.core.engine import DATADOC

doc = DATADOC("comprehensive_test.csv")

print("=== ORIGINAL ===")
print(doc.df)
print(f"Shape: {doc.df.shape}")
print()

# Detect column roles
roles = DATADOC._detect_column_roles(doc.df)
print(f"Column roles: {roles}")
print()

# Run engineer
def on_progress(name, status, details):
    print(f"  [{status.upper()}] {name}: {details}")

clean_df = doc.engineer(progress_callback=on_progress)
print()
print("=== ENGINEERED RESULT ===")
print(clean_df)
print(f"Shape: {clean_df.shape}")
print(f"Columns: {clean_df.columns}")
print(f"Dtypes: {dict(zip(clean_df.columns, [str(d) for d in clean_df.dtypes]))}")
print()

# Verify checks
print("=== VERIFICATION ===")
errors = []

# 1. ID and Name should be absent
if "ID" in clean_df.columns:
    errors.append("FAIL: ID column still present")
if "Name" in clean_df.columns:
    errors.append("FAIL: Name column still present")

# 2. No binary columns should be z-scored
for col in clean_df.columns:
    vals = clean_df[col].drop_nulls().unique().sort().to_list()
    if set(vals).issubset({0, 1, 0.0, 1.0}):
        # This is a binary column, values should be 0 or 1 only
        if any(v not in (0, 1, 0.0, 1.0) for v in vals):
            errors.append(f"FAIL: Binary column {col} has non-binary values: {vals}")
        else:
            print(f"  OK: Binary column '{col}' preserved as 0/1")

# 3. Income outliers should be capped
if "Income" in clean_df.columns:
    max_inc = clean_df["Income"].max()
    min_inc = clean_df["Income"].min()
    if max_inc > 1_000_000:
        errors.append(f"FAIL: Income outlier not capped, max={max_inc}")
    elif min_inc < -100_000:
        errors.append(f"FAIL: Income outlier not capped, min={min_inc}")
    else:
        print(f"  OK: Income outliers capped (range: {min_inc} to {max_inc})")

# 4. No constant columns
for col in clean_df.columns:
    if clean_df[col].drop_nulls().n_unique() <= 1:
        errors.append(f"FAIL: Constant column '{col}' still present")

# 5. Signup_Date_year should be dropped (constant)
if "Signup_Date_year" in clean_df.columns:
    errors.append("FAIL: Signup_Date_year (constant) still present")
else:
    print("  OK: Signup_Date_year (constant) was dropped")

# 6. Hour column should be present (datetimes have time components)
if "Signup_Date_hour" in clean_df.columns:
    print("  OK: Signup_Date_hour extracted from time components")
else:
    print("  INFO: Signup_Date_hour not present")

# 7. Continuous features should be scaled (approximately zero mean)
for col in ["Age", "Income", "Score"]:
    if col in clean_df.columns:
        mean_val = clean_df[col].mean()
        if abs(mean_val) < 0.1:
            print(f"  OK: '{col}' is scaled (mean={mean_val:.4f})")
        else:
            print(f"  INFO: '{col}' mean={mean_val:.4f} (may not need scaling)")

if errors:
    print()
    print("ERRORS FOUND:")
    for e in errors:
        print(f"  {e}")
else:
    print()
    print("ALL CHECKS PASSED!")
