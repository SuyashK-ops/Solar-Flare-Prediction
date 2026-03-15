from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .preprocessing import NUMERIC_FEATURE_CANDIDATES

sns.set_theme(style="whitegrid", context="talk")


def get_available_features(labeled_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column in labeled_df.columns and labeled_df[column].notna().any()
    ]


def add_temporal_features(
    labeled_df: pd.DataFrame,
    rolling_window: int = 3,
) -> pd.DataFrame:
    df = labeled_df.copy()
    if df.empty:
        return df

    df = df.sort_values(["noaa_active_region", "observation_time"]).reset_index(drop=True)
    base_features = get_available_features(df)

    for feature in base_features:
        grouped = df.groupby("noaa_active_region")[feature]
        df[f"{feature}_delta"] = grouped.diff()
        df[f"{feature}_rolling_mean"] = grouped.transform(
            lambda series: series.rolling(window=rolling_window, min_periods=1).mean()
        )
        df[f"{feature}_rolling_std"] = grouped.transform(
            lambda series: series.rolling(window=rolling_window, min_periods=1).std()
        )

    for column in df.columns:
        if column.endswith("_delta") or column.endswith("_rolling_std"):
            df[column] = df[column].fillna(0.0)

    return df


def add_strong_flare_label(labeled_df: pd.DataFrame, flare_df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    df = labeled_df.copy()
    if df.empty:
        df["strong_flare_next_24h"] = []
        return df

    strong_flare_df = flare_df[flare_df["flare_class"].str.startswith(("M", "X"), na=False)].copy()
    flare_times = strong_flare_df[["noaa_active_region", "event_start"]].dropna().copy()
    flare_times["event_start"] = pd.to_datetime(flare_times["event_start"], utc=True, errors="coerce")
    grouped_times = (
        flare_times.groupby("noaa_active_region")["event_start"]
        .apply(lambda series: series.sort_values().to_numpy(dtype="datetime64[ns]"))
        .to_dict()
    )

    window = pd.Timedelta(hours=window_hours).to_timedelta64()
    labels: list[int] = []
    for row in df.itertuples(index=False):
        events = grouped_times.get(row.noaa_active_region)
        if events is None or len(events) == 0:
            labels.append(0)
            continue
        current_time = row.observation_time.to_datetime64()
        future_events = events[events > current_time]
        if len(future_events) == 0:
            labels.append(0)
            continue
        labels.append(int((future_events[0] - current_time) <= window))

    df["strong_flare_next_24h"] = labels
    return df


def build_forecasting_dataset(labeled_df: pd.DataFrame, flare_df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(labeled_df)
    df = add_strong_flare_label(df, flare_df)
    return df


def time_based_split(
    forecasting_df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if forecasting_df.empty:
        raise ValueError("The forecasting dataset is empty.")

    df = forecasting_df.sort_values("observation_time").reset_index(drop=True)
    split_index = max(1, int(len(df) * train_fraction))
    split_index = min(split_index, len(df) - 1)
    split_time = df.loc[split_index, "observation_time"]

    train_df = df[df["observation_time"] < split_time].copy()
    test_df = df[df["observation_time"] >= split_time].copy()
    return train_df, test_df, split_time


def summarize_time_split(train_df: pd.DataFrame, test_df: pd.DataFrame, split_time: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subset": ["train", "test"],
            "rows": [len(train_df), len(test_df)],
            "positive_rate": [
                float(train_df["flare_next_24h"].mean()) if not train_df.empty else 0.0,
                float(test_df["flare_next_24h"].mean()) if not test_df.empty else 0.0,
            ],
            "strong_flare_rate": [
                float(train_df["strong_flare_next_24h"].mean()) if "strong_flare_next_24h" in train_df else 0.0,
                float(test_df["strong_flare_next_24h"].mean()) if "strong_flare_next_24h" in test_df else 0.0,
            ],
            "time_boundary_utc": [split_time, split_time],
        }
    )


def plot_label_rate_over_time(
    forecasting_df: pd.DataFrame,
    output_dir: str | Path,
    frequency: str = "1D",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rate_df = (
        forecasting_df.set_index("observation_time")["flare_next_24h"]
        .resample(frequency)
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=rate_df, x="observation_time", y="flare_next_24h", marker="o", color="#d95f02")
    plt.title("Daily Rate of flare_next_24h")
    plt.xlabel("Observation Time")
    plt.ylabel("Positive Rate")
    plt.tight_layout()

    destination = output_path / "daily_flare_label_rate.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def plot_feature_drift(
    forecasting_df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 4,
    frequency: str = "1D",
) -> Path:
    base_features = get_available_features(forecasting_df)[:top_n]
    if not base_features:
        raise ValueError("No usable features are available for the drift plot.")

    aggregated = (
        forecasting_df.set_index("observation_time")[base_features]
        .resample(frequency)
        .mean()
        .reset_index()
        .melt(id_vars="observation_time", var_name="feature", value_name="value")
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=aggregated, x="observation_time", y="value", hue="feature", marker="o")
    plt.title("Mean Feature Evolution Over Time")
    plt.xlabel("Observation Time")
    plt.ylabel("Mean Feature Value")
    plt.tight_layout()

    destination = output_path / "feature_drift_over_time.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination
