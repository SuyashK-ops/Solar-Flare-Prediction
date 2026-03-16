from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="talk")


def get_ml_feature_columns(forecasting_df: pd.DataFrame) -> list[str]:
    excluded = {
        "observation_time",
        "noaa_active_region",
        "flare_next_24h",
        "strong_flare_next_24h",
    }
    return [
        column
        for column in forecasting_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(forecasting_df[column])
        and forecasting_df[column].notna().any()
    ]


def prepare_ml_dataset(
    forecasting_df: pd.DataFrame,
    label_column: str = "flare_next_24h",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = forecasting_df.sort_values("observation_time").reset_index(drop=True).copy()
    feature_columns = get_ml_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for ML.")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' was not found.")

    X = df[feature_columns]
    y = df[label_column].astype(int)
    timestamps = df["observation_time"]
    return X, y, timestamps, feature_columns


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    split_index = max(1, int(len(X) * train_fraction))
    split_index = min(split_index, len(X) - 1)
    X_train, X_test = X.iloc[:split_index].copy(), X.iloc[split_index:].copy()
    y_train, y_test = y.iloc[:split_index].copy(), y.iloc[split_index:].copy()
    t_train, t_test = timestamps.iloc[:split_index].copy(), timestamps.iloc[split_index:].copy()
    return X_train, X_test, y_train, y_test, t_train, t_test


def _scaled_numeric_pipeline(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
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


def _tree_numeric_pipeline(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), feature_columns)
        ]
    )


def build_final_model_searches(feature_columns: list[str]) -> dict[str, GridSearchCV]:
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", _scaled_numeric_pipeline(feature_columns)),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", _tree_numeric_pipeline(feature_columns)),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    boosting_pipeline = Pipeline(
        steps=[
            ("preprocessor", _tree_numeric_pipeline(feature_columns)),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    time_cv = TimeSeriesSplit(n_splits=4)
    return {
        "Logistic Regression": GridSearchCV(
            logistic_pipeline,
            param_grid={"model__C": [0.1, 0.3, 1.0, 3.0]},
            scoring="average_precision",
            cv=time_cv,
            n_jobs=1,
        ),
        "Random Forest": GridSearchCV(
            forest_pipeline,
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, None],
            },
            scoring="average_precision",
            cv=time_cv,
            n_jobs=1,
        ),
        "HistGradientBoosting": GridSearchCV(
            boosting_pipeline,
            param_grid={
                "model__learning_rate": [0.03, 0.1],
                "model__max_iter": [100, 200],
                "model__max_depth": [3, 5],
            },
            scoring="average_precision",
            cv=time_cv,
            n_jobs=1,
        ),
    }


def train_final_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
) -> dict[str, GridSearchCV]:
    searches = build_final_model_searches(feature_columns)
    for search in searches.values():
        search.fit(X_train, y_train)
    return searches


def evaluate_final_models(
    searches: dict[str, GridSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for name, search in searches.items():
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        scores = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        row: dict[str, float | str] = {
            "model": name,
            "best_cv_score": float(search.best_score_),
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        row["average_precision"] = average_precision_score(y_test, scores) if scores is not None else float("nan")
        row["roc_auc"] = roc_auc_score(y_test, scores) if scores is not None else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("average_precision", ascending=False).reset_index(drop=True)


def plot_final_model_comparison(results_df: pd.DataFrame, output_dir: str | Path) -> Path:
    metric_frame = results_df.melt(
        id_vars="model",
        value_vars=["best_cv_score", "average_precision", "roc_auc", "f1"],
        var_name="metric",
        value_name="score",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=metric_frame, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Final ML Model Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=10)
    plt.tight_layout()

    destination = output_path / "final_ml_model_comparison.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def plot_pr_curves(
    searches: dict[str, GridSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, search in searches.items():
        model = search.best_estimator_
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
            PrecisionRecallDisplay.from_predictions(y_test, scores, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves on the Chronological Test Set")
    fig.tight_layout()

    destination = output_path / "final_ml_precision_recall_curves.png"
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def plot_roc_curves(
    searches: dict[str, GridSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, search in searches.items():
        model = search.best_estimator_
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=ax)
    ax.set_title("ROC Curves on the Chronological Test Set")
    fig.tight_layout()

    destination = output_path / "final_ml_roc_curves.png"
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def plot_permutation_importance(
    search: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    output_dir: str | Path,
    model_name: str,
    top_n: int = 10,
) -> Path:
    model = search.best_estimator_
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="average_precision",
    )
    importance_series = pd.Series(importance.importances_mean, index=feature_columns).sort_values(ascending=False).head(top_n)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_series.values, y=importance_series.index, color="#2a9d8f")
    plt.title(f"Permutation Importance: {model_name}")
    plt.xlabel("Decrease in Average Precision")
    plt.ylabel("")
    plt.tight_layout()

    destination = output_path / f"{model_name.lower().replace(' ', '_')}_permutation_importance.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination
