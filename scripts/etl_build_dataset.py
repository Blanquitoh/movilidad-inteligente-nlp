"""Build processed datasets from raw CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from src.utils.logger import logger
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    _SCRIPT_PARENT = Path(__file__).resolve().parents[1]
    _SCRIPT_PARENT_STR = str(_SCRIPT_PARENT)
    if _SCRIPT_PARENT_STR not in sys.path:
        sys.path.insert(0, _SCRIPT_PARENT_STR)

from scripts.bootstrap import bootstrap_project

_PROJECT_ROOT = bootstrap_project()

from src.utils.text_cleaning import clean_text, ensure_stopwords


PANAMA_CATEGORY_PRIORITY: Sequence[tuple[str, str]] = (
    ("isaccident", "accidente"),
    ("isobstacle", "obstaculo"),
    ("isdanger", "peligro"),
    ("isincident", "incidente"),
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing {} rows", len(df))
    stopwords = ensure_stopwords()
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(lambda text: clean_text(text, stopwords))
    df["category"] = df["category"].astype(str)
    return df[["clean_text", "category", "text", "created_at", "location"]]


def build_datasets(config: dict) -> None:
    raw_source = config["paths"]["raw_data"]
    df = load_raw_data(raw_source)
    df = normalise_raw_dataframe(df)
    df = preprocess(df)

    split_cfg = config["split"]
    stratify_labels: pd.Series | None = df["category"]
    class_counts = df["category"].value_counts()
    test_size = split_cfg["test_size"]
    if isinstance(test_size, float):
        test_count = max(int(round(len(df) * test_size)), 1)
    else:
        test_count = int(test_size)

    if (class_counts < 2).any() or test_count < len(class_counts):
        logger.warning(
            "Insufficient samples per class for stratified split; falling back to random split"
        )
        stratify_labels = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=split_cfg.get("random_state", 42),
        stratify=stratify_labels,
    )

    train_path = Path(config["paths"]["processed_train"])
    test_path = Path(config["paths"]["processed_test"])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving train dataset to {}", train_path)
    train_df.to_csv(train_path, index=False)
    logger.info("Saving test dataset to {}", test_path)
    test_df.to_csv(test_path, index=False)


def expand_raw_sources(raw_source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve a raw data source into a list of CSV file paths."""
    if isinstance(raw_source, (str, Path)):
        candidates = [raw_source]
    else:
        candidates = list(raw_source)

    resolved: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            files = sorted(p for p in path.glob("*.csv") if "stop_word" not in p.name.lower())
            resolved.extend(files)
        elif path.is_file():
            resolved.append(path)
        else:
            if any(token in path.as_posix() for token in ("*", "?")):
                resolved.extend(sorted(path.parent.glob(path.name)))
            else:
                logger.warning("Raw data source {} not found; skipping", path)
    return resolved


def load_raw_data(raw_source: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Load raw data from one or many CSV files."""
    csv_files = expand_raw_sources(raw_source)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for raw data source: {raw_source}")

    logger.info("Loading raw data from {} file(s)", len(csv_files))
    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        logger.debug("Reading raw data file {}", csv_file)
        frames.append(pd.read_csv(csv_file))

    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def derive_category_from_flags(row: pd.Series) -> str:
    """Map boolean incident flags into a human-readable category."""
    for column, label in PANAMA_CATEGORY_PRIORITY:
        if column in row:
            value = row[column]
            if pd.notna(value) and bool(value):
                return label
    return "informativo"


def normalise_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise varying raw schemas into the canonical training schema."""
    expected_columns = ["text", "category", "created_at", "location"]
    if set(expected_columns).issubset(df.columns):
        logger.info("Raw data already in canonical schema")
        return df[expected_columns].copy()

    panama_columns = {"tweet_text", "tweet_created"}
    if panama_columns.issubset(df.columns):
        logger.info("Normalising Panama multi-file dataset")
        location_series = df.get("identified_place")
        if location_series is None:
            location_series = pd.Series([""] * len(df))

        normalised = pd.DataFrame(
            {
                "text": df["tweet_text"].astype(str).str.strip(),
                "created_at": df["tweet_created"].astype(str),
                "location": location_series.fillna("").astype(str),
            }
        )
        normalised["category"] = df.apply(derive_category_from_flags, axis=1)

        normalised = normalised[normalised["text"].astype(bool)]
        normalised = normalised.drop_duplicates(subset=["text", "category"])
        return normalised[["text", "category", "created_at", "location"]]

    raise ValueError("Unsupported raw data schema; expected canonical columns or Panama dataset format")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets for training")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    build_datasets(config)


if __name__ == "__main__":
    main()
