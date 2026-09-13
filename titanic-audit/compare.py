"""Honest before/after: fixed stratified split, leakage-safe DATADOC fit on train only.

Baseline = typical Kaggle-starter manual prep on the same split.
DATADOC   = datadoc fit (train only) + transform, same model, same split.
Usage: python titanic-audit/compare.py [--preset tree] [--rare 0.02] [--clip/--no-clip]
"""

import argparse
import sys

import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from datadoc import DataDocPipeline, PipelineConfig


def manual_baseline(train: pl.DataFrame, test: pl.DataFrame):
    """Typical hand-written Kaggle starter prep (fit on train, apply to test)."""
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    cat_cols = ["Sex", "Embarked"]
    medians = {c: train[c].median() for c in num_cols}
    modes = {c: train[c].drop_nulls().mode()[0] for c in cat_cols}

    def prep(df: pl.DataFrame) -> pl.DataFrame:
        out = df.clone()
        for c in num_cols:
            out = out.with_columns(pl.col(c).fill_null(medians[c]))
        for c in cat_cols:
            out = out.with_columns(pl.col(c).fill_null(modes[c]))
        return out.select(num_cols + cat_cols).to_dummies(columns=cat_cols, drop_first=True)

    ptr, pte = prep(train), prep(test)
    # align dummy columns (unseen categories -> zeros, like a careful manual pipeline)
    for c in ptr.columns:
        if c not in pte.columns:
            pte = pte.with_columns(pl.lit(0, dtype=pl.UInt8).alias(c))
    pte = pte.select(ptr.columns)
    return ptr.to_numpy(), train["Survived"].to_numpy(), pte.to_numpy(), test["Survived"].to_numpy()


def datadoc_prep(train: pl.DataFrame, test: pl.DataFrame, args):
    ids = ["PassengerId"] if args.keep_ticket else ["PassengerId", "Ticket"]
    cfg = PipelineConfig(
        target="Survived",
        task="classification",
        drop_identifiers=True,
        identifier_columns=ids,
        scaling="none" if args.preset == "tree" else "standard",
        clip_outliers=args.clip,
        rare_category_min_frequency=args.rare,
        estimator_family="tree" if args.preset == "tree" else "linear",
    )
    pipe = DataDocPipeline(cfg).fit(train)
    tr = pipe.transform(train).drop("Survived").to_numpy()
    te = pipe.transform(test).drop("Survived").to_numpy()
    # keep only numeric (pipeline output is all-numeric here, but be safe)
    return tr, train["Survived"].to_numpy(), te, test["Survived"].to_numpy(), pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tree")
    ap.add_argument("--rare", type=float, default=0.02)
    ap.add_argument("--clip", action="store_true", default=False)
    ap.add_argument("--keep-ticket", action="store_true", default=False)
    args = ap.parse_args()

    df = pl.read_csv("titanic.csv")
    idx_tr, idx_te = train_test_split(
        range(df.height), test_size=0.2, random_state=42, stratify=df["Survived"].to_list()
    )
    train, test = df[idx_tr], df[idx_te]
    print(f"split: train={train.height} test={test.height} seed=42 stratified")

    for model_name, make in [
        ("RandomForest(300)", lambda: RandomForestClassifier(n_estimators=300, random_state=42)),
        ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(random_state=42)),
        ("LogisticRegression", lambda: LogisticRegression(max_iter=2000)),
    ]:
        Xtr, ytr, Xte, yte = manual_baseline(train, test)
        model = make().fit(Xtr, ytr)
        base = accuracy_score(yte, model.predict(Xte))
        base_bal = balanced_accuracy_score(yte, model.predict(Xte))

        Xtr2, ytr2, Xte2, yte2, pipe = datadoc_prep(train, test, args)
        model2 = make().fit(Xtr2, ytr2)
        model2.fit(Xtr2, ytr2)
        dd = accuracy_score(yte2, model2.predict(Xte2))
        dd_bal = balanced_accuracy_score(yte2, model2.predict(Xte2))
        print(
            f"{model_name}: baseline acc={base:.4f} bal={base_bal:.4f} "
            f"| DATADOC acc={dd:.4f} bal={dd_bal:.4f} "
            f"| delta={dd - base:+.4f} (features {Xtr.shape[1]} -> {Xtr2.shape[1]})"
        )


if __name__ == "__main__":
    main()
