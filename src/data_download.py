from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import pandas as pd
import requests

NOAA_EVENT_REPORT_URL = (
    "ftp://ftp.swpc.noaa.gov/pub/warehouse/{year}/{year}_events/{day}events.txt"
)

SHARP_KEYWORDS = [
    "T_REC",
    "NOAA_ARS",
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


@dataclass
class DownloadResult:
    flare_files: list[Path]
    active_region_file: Path | None


def ensure_project_dirs(project_root: str | Path) -> None:
    root = Path(project_root)
    for name in ("data", "figures", "notebooks", "src"):
        (root / name).mkdir(parents=True, exist_ok=True)


def download_noaa_flare_reports(
    start_date: str | date,
    end_date: str | date,
    output_dir: str | Path,
    timeout: int = 60,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        day = current_date.strftime("%Y%m%d")
        url = NOAA_EVENT_REPORT_URL.format(year=year, day=day)
        destination = output_path / f"{day}_events.txt"
        with urlopen(url, timeout=timeout) as response:
            destination.write_bytes(response.read())
        saved_files.append(destination)
        current_date += timedelta(days=1)

    return saved_files


def download_sharp_parameters(
    start_time: str,
    end_time: str,
    output_path: str | Path,
    cadence: str = "12h",
    series: str = "hmi.sharp_cea_720s",
) -> Path:
    try:
        import drms
    except ImportError as exc:
        raise ImportError(
            "The 'drms' package is required to query SHARP parameters. "
            "Install it with: pip install drms"
        ) from exc

    # SHARP series use HARPNUM as the first prime key and T_REC as the second.
    # Leaving the first selector empty requests all active regions over the time range.
    query = f"{series}[][{start_time}-{end_time}@{cadence}]"
    client = drms.Client()
    frame = client.query(query, key=",".join(SHARP_KEYWORDS))
    if frame is None or frame.empty:
        raise ValueError("No SHARP records were returned for the requested interval.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def download_stage_one_data(
    project_root: str | Path,
    start_date: str | date,
    end_date: str | date,
    start_time: str,
    end_time: str,
    cadence: str = "12h",
) -> DownloadResult:
    root = Path(project_root)
    data_dir = root / "data"
    flare_dir = data_dir / "noaa_flare_reports"
    flare_files = download_noaa_flare_reports(
        start_date=start_date,
        end_date=end_date,
        output_dir=flare_dir,
    )
    sharp_file = download_sharp_parameters(
        start_time=start_time,
        end_time=end_time,
        cadence=cadence,
        output_path=data_dir / "sharp_parameters.csv",
    )
    return DownloadResult(flare_files=flare_files, active_region_file=sharp_file)


def load_csv_or_parquet(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)
