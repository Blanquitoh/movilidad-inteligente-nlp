"""Evaluate the trained text classifier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix

if __package__ is None or __package__ == "":
    _SCRIPT_PARENT = Path(__file__).resolve().parents[1]
    _SCRIPT_PARENT_STR = str(_SCRIPT_PARENT)
    if _SCRIPT_PARENT_STR not in sys.path:
        sys.path.insert(0, _SCRIPT_PARENT_STR)

from scripts.bootstrap import bootstrap_project

_PROJECT_ROOT = bootstrap_project()

from src.utils.logger import logger
from src.infrastructure.nlp.model_builder import TextClassifierPipeline


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the text classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser.parse_args()


def load_model(model_path: Path) -> TextClassifierPipeline:
    if model_path.exists():
        return TextClassifierPipeline.load(model_path)
    base = model_path.with_suffix("")
    if base.with_suffix(".h5").exists():
        return TextClassifierPipeline.load(base)
    raise FileNotFoundError(f"Model artifact not found at {model_path}")


def plot_confusion_matrix(cm, labels, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Real",
        xlabel="Predicho",
        title="Matriz de confusión",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i, j in ((i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    test_path = Path(config["paths"]["processed_test"])
    model_path = Path(config["paths"]["model_artifact"])

    logger.info("Loading test data from {}", test_path)
    df = pd.read_csv(test_path)
    texts = df["clean_text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()

    classifier = load_model(model_path)
    predictions = classifier.predict(texts)

    report = classification_report(labels, predictions)
    logger.info("Evaluation report:\n{}", report)

    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, sorted(set(labels)), Path("reports/figures/confusion_matrix.png"))


if __name__ == "__main__":
    main()
