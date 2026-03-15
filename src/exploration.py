from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from .preprocessing import NUMERIC_FEATURE_CANDIDATES

sns.set_theme(style="whitegrid", context="talk")


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the requested columns exist: {candidates}")


def summarize_dataset(flare_df: pd.DataFrame, active_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "flare_events",
                "active_region_snapshots",
                "unique_active_regions",
                "positive_labels",
                "positive_rate",
            ],
            "value": [
                len(flare_df),
                len(active_df),
                labeled_df["noaa_active_region"].nunique() if not labeled_df.empty else 0,
                int(labeled_df["flare_next_24h"].sum()) if "flare_next_24h" in labeled_df else 0,
                float(labeled_df["flare_next_24h"].mean()) if "flare_next_24h" in labeled_df else 0.0,
            ],
        }
    )


def compute_pearson_correlations(labeled_df: pd.DataFrame) -> pd.Series:
    numeric_columns = [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column in labeled_df.columns and labeled_df[column].notna().any()
    ]
    if not numeric_columns:
        return pd.Series(dtype=float)
    correlations = labeled_df[numeric_columns + ["flare_next_24h"]].corr(method="pearson")
    return correlations["flare_next_24h"].drop("flare_next_24h").sort_values(ascending=False)


def compute_mutual_information(labeled_df: pd.DataFrame) -> pd.Series:
    numeric_columns = [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column in labeled_df.columns and labeled_df[column].notna().any()
    ]
    if not numeric_columns:
        return pd.Series(dtype=float)
    feature_frame = labeled_df[numeric_columns].fillna(labeled_df[numeric_columns].median())
    target = labeled_df["flare_next_24h"].astype(int)
    values = mutual_info_classif(feature_frame, target, random_state=42)
    return pd.Series(values, index=numeric_columns).sort_values(ascending=False)


def generate_eda_figures(
    flare_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    magnetic_flux_col = _pick_existing_column(labeled_df, ["USFLUX", "TOTBSQ", "MEANGBT"])
    magnetic_strength_col = _pick_existing_column(labeled_df, ["MEANGBT", "MEANGBZ", "MEANGBH", "TOTBSQ"])
    area_col = _pick_existing_column(labeled_df, ["AREA_ACR"])
    numeric_columns = [column for column in NUMERIC_FEATURE_CANDIDATES if column in labeled_df.columns]

    plt.figure(figsize=(10, 6))
    flare_df["flare_class"].str[0].value_counts().sort_index().plot(kind="bar", color="#d95f02")
    plt.title("Histogram of Flare Classes")
    plt.xlabel("Flare Class")
    plt.ylabel("Count")
    path = figure_dir / "flare_class_histogram.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    figure_paths["flare_class_histogram"] = path

    plt.figure(figsize=(10, 6))
    sns.histplot(labeled_df[magnetic_strength_col].dropna(), bins=30, kde=True, color="#1b9e77")
    plt.title("Distribution of Magnetic Field Strength")
    plt.xlabel(magnetic_strength_col)
    path = figure_dir / "magnetic_field_strength_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    figure_paths["magnetic_field_strength_distribution"] = path

    plt.figure(figsize=(10, 6))
    sns.histplot(labeled_df[area_col].dropna(), bins=30, kde=True, color="#7570b3")
    plt.title("Distribution of Active Region Area")
    plt.xlabel(area_col)
    path = figure_dir / "active_region_area_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    figure_paths["active_region_area_distribution"] = path

    if numeric_columns:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            labeled_df[numeric_columns].corr(),
            cmap="coolwarm",
            center=0,
            annot=False,
            square=True,
        )
        plt.title("Correlation Heatmap of Physical Parameters")
        path = figure_dir / "parameter_correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths["parameter_correlation_heatmap"] = path

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=labeled_df,
        x=magnetic_flux_col,
        y="flare_next_24h",
        alpha=0.7,
        color="#e7298a",
    )
    plt.title("Magnetic Flux vs Flare Occurrence")
    plt.xlabel(magnetic_flux_col)
    plt.ylabel("flare_next_24h")
    path = figure_dir / "magnetic_flux_vs_flare_occurrence.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    figure_paths["magnetic_flux_vs_flare_occurrence"] = path

    return figure_paths

