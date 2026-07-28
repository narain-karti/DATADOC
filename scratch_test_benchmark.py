import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

print("1️⃣ Generating a highly realistic dirty dataset...")
np.random.seed(42)
n_samples = 5000

# True Clean Features
age_clean = np.random.normal(40, 15, n_samples)
income_clean = np.random.normal(60000, 20000, n_samples)
credit_score_clean = np.random.normal(650, 50, n_samples)

# Target Variable (Calculated strictly from CLEAN data)
spending = (age_clean * 10) + (income_clean * 0.05) + (credit_score_clean * 2) + np.random.normal(0, 500, n_samples)

# Now we corrupt the features to create the "Dirty" dataset
age = age_clean.copy()
income = income_clean.copy()
credit_score = credit_score_clean.copy()

# Inject Extreme Outliers (Noise that ruins the relationship)
outlier_idx = np.random.choice(n_samples, 200, replace=False)
income[outlier_idx[:100]] = 9999999
age[outlier_idx[100:]] = 150

# Inject Missing Values (NaNs)
nan_idx = np.random.choice(n_samples, 600, replace=False)
credit_score[nan_idx[:400]] = np.nan
age[nan_idx[400:]] = np.nan

# Categorical String Feature (Provides a slight signal)
cities = ["New York", "London", "Tokyo", "Paris"]
city_col = np.random.choice(cities, n_samples)
# Let's say New Yorkers spend $1000 more
spending += np.where(city_col == "New York", 1000, 0)

# Useless constant column
constant_col = ["Constant"] * n_samples

df_features = pd.DataFrame({
    'ID': range(n_samples),
    'Age': age,
    'Income': income,
    'Credit_Score': credit_score,
    'City': city_col,
    'Useless_Col': constant_col
})

df_target = pd.Series(spending, name='Target_Spending')

# Save just the features for DATADOC to clean
df_features.to_csv('dirty_features.csv', index=False)

# EXPERIMENT 1: MANUAL QUICK FIXES
print("\n❌ EXPERIMENT 1: Training on dirty data...")
df_manual = df_features.drop(columns=['City', 'Useless_Col', 'ID']) 
df_manual = df_manual.fillna(0) # Terrible idea, but standard lazy fix

X_train_dirty, X_test_dirty, y_train, y_test = train_test_split(df_manual, df_target, test_size=0.2, random_state=42)

rf_dirty = RandomForestRegressor(random_state=42, n_estimators=50)
rf_dirty.fit(X_train_dirty, y_train)
dirty_preds = rf_dirty.predict(X_test_dirty)
dirty_r2 = r2_score(y_test, dirty_preds)
print(f"Dirty R² Score: {dirty_r2:.4f}")

# EXPERIMENT 2: DATADOC
print("\n✅ EXPERIMENT 2: Training on DATADOC engineered data...")
from datadoc.core.engine import DATADOC

doc = DATADOC('dirty_features.csv')
# Monkeypatch to avoid stdout spam if we want, but it's fine
X_clean = doc.engineer().to_pandas()

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, df_target, test_size=0.2, random_state=42)

rf_clean = RandomForestRegressor(random_state=42, n_estimators=50)
rf_clean.fit(X_train_clean, y_train_clean)
clean_preds = rf_clean.predict(X_test_clean)
clean_r2 = r2_score(y_test_clean, clean_preds)
print(f"DATADOC R² Score: {clean_r2:.4f}")

print("\n🏆 FINAL BENCHMARK RESULTS")
print(f"Dirty Data R² Score:   {dirty_r2:.4f}")
print(f"DATADOC R² Score:      {clean_r2:.4f}")
improvement = ((clean_r2 - dirty_r2) / abs(dirty_r2)) * 100
print(f"DATADOC Performance Boost: +{improvement:.2f}%")
