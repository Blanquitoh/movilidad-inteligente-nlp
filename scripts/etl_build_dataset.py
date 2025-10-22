"""Build processed datasets from raw CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from src.utils.text_cleaning import clean_text, ensure_stopwords


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
    raw_path = Path(config["paths"]["raw_data"])
    train_path = Path(config["paths"]["processed_train"])
    test_path = Path(config["paths"]["processed_test"])

    logger.info("Loading raw data from {}", raw_path)
    df = pd.read_csv(raw_path)
    df = preprocess(df)

    split_cfg = config["split"]
    train_df, test_df = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg.get("random_state", 42),
        stratify=df["category"],
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving train dataset to {}", train_path)
    train_df.to_csv(train_path, index=False)
    logger.info("Saving test dataset to {}", test_path)
    test_df.to_csv(test_path, index=False)


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
