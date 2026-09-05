from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.evaluate import business_metrics, evaluate_classifier
from ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN
from ml.generate_dataset import generate_population

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(exist_ok=True)


def _customer_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["customer_id"]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=7)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["customer_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def train_logistic_regression(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Pipeline:
    numeric_features = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", numeric_features),
        ]
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return pipeline


def train_catboost(train_df: pd.DataFrame, val_df: pd.DataFrame) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.05,
        depth=8,
        iterations=700,
        random_seed=42,
        verbose=False,
        cat_features=CATEGORICAL_FEATURES,
        early_stopping_rounds=50,
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        eval_set=(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]),
        use_best_model=True,
    )
    return model


def evaluate_models(
    name: str,
    model,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    score = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    metrics = evaluate_classifier(test_df[TARGET_COLUMN], score, threshold=0.5)
    business = business_metrics(test_df, score, threshold=0.5)
    metrics["business"] = business
    metrics["model_name"] = name
    return metrics


def main() -> None:
    df = generate_population(n_cases=50000, seeds=[7, 11, 13, 17, 19, 23], customer_count=3000)
    train_df, val_df, test_df = _customer_split(df)

    logistic_model = train_logistic_regression(train_df, val_df)
    cat_model = train_catboost(train_df, val_df)

    cat_model.save_model(str(MODEL_DIR / "catboost_recovery_model.cbm"))
    joblib.dump(logistic_model, str(MODEL_DIR / "logistic_recovery_model.joblib"))

    logistic_metrics = evaluate_models("LogisticRegression", logistic_model, test_df)
    cat_metrics = evaluate_models("CatBoostClassifier", cat_model, test_df)

    print("=== Recovery Model Comparison ===")
    print(f"LogisticRegression: ROC-AUC={logistic_metrics['roc_auc']:.4f}, PR-AUC={logistic_metrics['pr_auc']:.4f}, precision={logistic_metrics['precision']:.4f}, recall={logistic_metrics['recall']:.4f}, Brier={logistic_metrics['brier_score']:.4f}")
    print(f"CatBoostClassifier: ROC-AUC={cat_metrics['roc_auc']:.4f}, PR-AUC={cat_metrics['pr_auc']:.4f}, precision={cat_metrics['precision']:.4f}, recall={cat_metrics['recall']:.4f}, Brier={cat_metrics['brier_score']:.4f}")
    print("Business impact (threshold=0.5):")
    print(f"  LogisticRegression: {logistic_metrics['business']}")
    print(f"  CatBoostClassifier: {cat_metrics['business']}")


if __name__ == "__main__":
    main()
