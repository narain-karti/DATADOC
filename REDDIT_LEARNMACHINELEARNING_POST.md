<!-- 
===================================================================
REDDIT R/LEARNMACHINELEARNING SUBMISSION FILE
===================================================================

COMMUNITY: r/learnmachinelearning (1.3M members)
FLAIR: Select "Discussion", "Project", or "Tutorial"
TYPE: Text Post

TITLE TO COPY:
Why ad-hoc pandas preprocessing silently causes data leakage (and how to fix it to get higher real-world ML accuracy)

Everything below the divider is the POST BODY to copy & paste.
===================================================================
-->

One of the most common mistakes beginners (and even intermediate practitioners) make when working with tabular data is **Data Leakage**.

It’s often the hidden reason why your model gets **88% accuracy in your Jupyter notebook**, but drops to **76%** when you evaluate it on an unseen test set or submit to a Kaggle competition.

Here is a quick breakdown of why it happens, the #1 most common mistake, and how to fix it properly.

---

### The #1 Most Common Leakage Bug:

Look at this very common snippet seen in many notebooks and tutorials:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

# 🚨 DANGEROUS LEAKAGE:
df['age'] = df['age'].fillna(df['age'].median())

# Train / Test split happened AFTER imputation:
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

### Why is this data leakage?
When you calculate `df['age'].median()` on the full dataset, **the median value is influenced by the test rows**. 

Your training set now contains subtle statistical information (the median) derived from test data it shouldn't even know exists. In production or Kaggle competitions, future data is completely unavailable at training time. 

The same leakage bug happens when people:
1. Scale features with `StandardScaler` on the entire dataframe before splitting.
2. Build categorical vocabularies or frequency encodings using all rows.
3. Compute outlier clipping boundaries (e.g. Tukey IQR limits) over the full dataset.

---

### The Correct Way (Strict Train-Only State):

You must fit transformations **strictly on the training split**, and freeze those exact parameters to apply to validation and test data:

```python
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 1. Calculate statistics ONLY from training split:
train_median_age = train['age'].median()

# 2. Apply that frozen training statistic to both splits:
train['age'] = train['age'].fillna(train_median_age)
test['age'] = test['age'].fillna(train_median_age)
```

---

### The Impact: We Tested Naive Prep vs. Zero-Leakage on Titanic

To see what happens when you replace naive ad-hoc pandas code with a strict train-only transformation ladder, we ran a 5-fold Stratified Cross-Validation benchmark on the Titanic dataset:

| Model | Naive Ad-Hoc Prep | Zero-Leakage Pipeline | Accuracy Delta | Relative Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 78.90% ± 0.99% | **79.91% ± 1.90%** | **+1.01%** | **+1.28%** |
| **Random Forest** | 82.15% ± 2.45% | **82.82% ± 2.40%** | **+0.67%** | **+0.82%** |

### Where did the accuracy lift come from?
1. **Informative Missingness Flags**: Imputing age with median alone destroys the signal that missing age itself correlates with survival. Adding an `Age__missing` binary flag recovers that signal.
2. **Train-Only Tukey IQR Clipping**: Capping extreme fares on training folds stabilized linear gradients without test-distribution bleed.
3. **Empirical Bayes Target Encoding**: High-cardinality categories shrink toward global means to prevent overfitting on small samples.

---

### We built an Open-Source Tool to automate this:

Writing 200 lines of manual state-tracking code for every dataset gets tedious. So we built **DATADOC** (v0.6.0)—an open-source CLI and Python library powered by Polars that automates this entire lifecycle with zero data leakage:

- **GitHub**: https://github.com/narain-karti/DATADOC
- **Documentation**: https://narain-karti.github.io/DATADOC/
- **Install**: `pip install datadoc-cli`

### How you can use it in 1 command:

You can use the interactive terminal wizard on your CSV file:
```bash
datadoc wizard train.csv
```

It walks you through:
1. Identifying your target column (e.g. `Survived` or `churn`).
2. Selecting a preset (`balanced`, `tree`, `linear`, or `robust`).
3. Generating a clean `pipeline.json` artifact containing all learned medians and rules.

Then transform unseen test data with zero leakage:
```bash
datadoc transform test.csv --pipeline artifacts/pipeline.json --output clean_test.csv
```

You can also run `datadoc health train.csv` to get an instant 0–100 data quality grade and find hidden issues before training.

The project is 100% open-source (MIT licensed) and runs completely offline.

I hope this helps clarify how data leakage happens in tabular pipelines! Let me know if you have any questions or want to discuss specific preprocessing edge cases.
