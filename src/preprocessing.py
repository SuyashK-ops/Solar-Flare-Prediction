from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NUMERIC_FEATURE_CANDIDATES = [
    "USFLUX",
    "AREA_ACR",
    "TOTUSJH",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "R_VALUE",
    "TOTBSQ",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
]


def _combine_datetime(date_token: str, time_token: str) -> pd.Timestamp:
    cleaned_time = re.sub(r"[^0-9]", "", str(time_token))
    if cleaned_time in {"", "////"}:
        return pd.NaT
    return pd.to_datetime(f"{date_token} {cleaned_time}", format="%Y %m %d %H%M", utc=True)


def _extract_region_id(value: object) -> int | float:
    if pd.isna(value):
        return np.nan
    matches = re.findall(r"\d{4,6}", str(value))
    if not matches:
        return np.nan
    return _normalize_region_id(int(matches[0]))


def _extract_region_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    matches = re.findall(r"\d{4,6}", str(value))
    return [_normalize_region_id(int(match)) for match in matches]


def _normalize_region_id(region_id: int) -> int:
    # Modern NOAA region identifiers are often stored as five digits with a
    # leading "1" in SHARP metadata (e.g. 14366), while flare reports use 4366.
    if region_id >= 10000:
        return region_id - 10000
    return region_id


def parse_noaa_flare_report(file_path: str | Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    file_date: str | None = None
    with Path(file_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(":Date:"):
                date_tokens = line.split(":", maxsplit=2)[-1].strip().split()
                if len(date_tokens) == 3:
                    file_date = " ".join(date_tokens)
                continue
            if not file_date or not line or line.startswith("#") or line.startswith(":"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            if parts[1] == "+":
                parts.pop(1)
            if len(parts) < 9 or parts[6] != "XRA":
                continue

            region = int(parts[-1]) if re.fullmatch(r"\d{4,6}", parts[-1]) else np.nan
            flare_class = parts[8]
            if not re.fullmatch(r"[A-Z]\d+\.\d", flare_class):
                continue

            records.append(
                {
                    "event_start": _combine_datetime(file_date, parts[1]),
                    "event_peak": _combine_datetime(file_date, parts[2]),
                    "event_end": _combine_datetime(file_date, parts[3]),
                    "flare_class": flare_class,
                    "noaa_active_region": region,
                }
            )

    return pd.DataFrame.from_records(records)


def load_flare_catalog(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames: list[pd.DataFrame] = []
    for path in paths:
        current_path = Path(path)
        if current_path.suffix.lower() == ".csv":
            frame = pd.read_csv(current_path)
        else:
            frame = parse_noaa_flare_report(current_path)
        frames.append(frame)

    flare_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return clean_flare_catalog(flare_df)


def clean_flare_catalog(flare_df: pd.DataFrame) -> pd.DataFrame:
    df = flare_df.copy()
    if df.empty:
        return df

    datetime_columns = [column for column in ("event_start", "event_peak", "event_end", "timestamp") if column in df.columns]
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    if "timestamp" in df.columns and "event_start" not in df.columns:
        df = df.rename(columns={"timestamp": "event_start"})

    if "active_region" in df.columns and "noaa_active_region" not in df.columns:
        df = df.rename(columns={"active_region": "noaa_active_region"})

    df["noaa_active_region"] = df["noaa_active_region"].map(_extract_region_id)
    df["flare_class"] = df["flare_class"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["event_start", "flare_class", "noaa_active_region"])
    df["noaa_active_region"] = df["noaa_active_region"].astype(int)
    df = df.sort_values("event_start").drop_duplicates(
        subset=["event_start", "flare_class", "noaa_active_region"]
    )
    return df.reset_index(drop=True)


def load_active_region_parameters(file_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    return clean_active_region_parameters(frame)


def clean_active_region_parameters(active_df: pd.DataFrame) -> pd.DataFrame:
    df = active_df.copy()
    if df.empty:
        return df

    timestamp_column = "T_REC" if "T_REC" in df.columns else "timestamp"
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        format="%Y.%m.%d_%H:%M:%S_TAI",
        utc=True,
        errors="coerce",
    )
    df = df.rename(columns={timestamp_column: "observation_time"})

    region_column = "NOAA_ARS" if "NOAA_ARS" in df.columns else "noaa_active_region"
    df[region_column] = df[region_column].map(_extract_region_ids)
    df = df.rename(columns={region_column: "noaa_active_region"})
    df = df.explode("noaa_active_region")

    numeric_columns = [column for column in NUMERIC_FEATURE_CANDIDATES if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    numeric_columns = [column for column in numeric_columns if df[column].notna().any()]

    df = df.dropna(subset=["observation_time", "noaa_active_region"])
    df["noaa_active_region"] = df["noaa_active_region"].astype(int)

    if df.empty:
        return df

    if numeric_columns:
        grouped = df.groupby("noaa_active_region")[numeric_columns]
        df[numeric_columns] = grouped.transform(lambda series: series.fillna(series.median()))
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    df = df.sort_values(["noaa_active_region", "observation_time"]).reset_index(drop=True)
    return df


def create_flare_next_24h_label(
    active_df: pd.DataFrame,
    flare_df: pd.DataFrame,
    window_hours: int = 24,
) -> pd.DataFrame:
    if active_df.empty:
        return active_df.copy()

    df = active_df.copy()
    flare_times = flare_df[["noaa_active_region", "event_start"]].dropna().copy()
    flare_times["event_start"] = pd.to_datetime(flare_times["event_start"], utc=True, errors="coerce")
    grouped_times = (
        flare_times.groupby("noaa_active_region")["event_start"]
        .apply(lambda series: np.sort(series.to_numpy(dtype="datetime64[ns]")))
        .to_dict()
    )

    window = np.timedelta64(window_hours, "h")
    labels: list[int] = []
    for row in df.itertuples(index=False):
        events = grouped_times.get(row.noaa_active_region)
        if events is None or len(events) == 0:
            labels.append(0)
            continue
        current_time = np.datetime64(row.observation_time.to_datetime64())
        future_events = events[events > current_time]
        if len(future_events) == 0:
            labels.append(0)
            continue
        labels.append(int((future_events[0] - current_time) <= window))

    df["flare_next_24h"] = labels
    return df


def prepare_modeling_dataset(
    flare_paths: str | Path | Iterable[str | Path],
    active_region_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flare_df = load_flare_catalog(flare_paths)
    active_df = load_active_region_parameters(active_region_path)
    labeled_df = create_flare_next_24h_label(active_df, flare_df)
    return flare_df, active_df, labeled_df
