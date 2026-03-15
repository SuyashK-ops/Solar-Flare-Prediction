from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import NUMERIC_FEATURE_CANDIDATES

sns.set_theme(style="whitegrid", context="talk")


def get_feature_columns(labeled_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column in labeled_df.columns and labeled_df[column].notna().any()
    ]


def build_modeling_table(labeled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = get_feature_columns(labeled_df)
    if not feature_columns:
        raise ValueError("No usable numeric feature columns were found.")
    if "flare_next_24h" not in labeled_df.columns:
        raise ValueError("The label column 'flare_next_24h' is missing.")

    modeling_df = labeled_df[feature_columns + ["flare_next_24h"]].dropna(subset=["flare_next_24h"]).copy()
    return modeling_df[feature_columns], modeling_df["flare_next_24h"].astype(int), feature_columns


def split_features_and_labels(
    labeled_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    X, y, feature_columns = build_modeling_table(labeled_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_columns


def build_baseline_models(feature_columns: list[str]) -> dict[str, Pipeline]:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), feature_columns)
        ]
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
) -> dict[str, Pipeline]:
    models = build_baseline_models(feature_columns)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = None

        row: dict[str, float | str] = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        row["average_precision"] = average_precision_score(y_test, scores) if scores is not None else float("nan")
        row["roc_auc"] = roc_auc_score(y_test, scores) if scores is not None else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def plot_model_comparison(results_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_frame = results_df.melt(
        id_vars="model",
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=metric_frame, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Baseline Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.xticks(rotation=10)
    plt.tight_layout()

    destination = output_path / "baseline_model_comparison.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def plot_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
    model_name: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()

    destination = output_path / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def plot_feature_importance(
    model: Pipeline,
    feature_columns: list[str],
    output_dir: str | Path,
    model_name: str,
    top_n: int = 10,
) -> Path:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = pd.Series(estimator.feature_importances_, index=feature_columns)
    elif hasattr(estimator, "coef_"):
        importance = pd.Series(estimator.coef_.ravel(), index=feature_columns).abs()
    else:
        raise ValueError(f"Model '{model_name}' does not expose feature importance values.")

    importance = importance.sort_values(ascending=False).head(top_n)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index, color="#4c78a8")
    plt.title(f"Top Feature Importance: {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()

    destination = output_path / f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination
