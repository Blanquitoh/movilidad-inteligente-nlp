"""Tests for the traffic EDA infrastructure helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure

from src.infrastructure.reports.traffic_eda import (
    FileSystemReportRepository,
    SimpleNotebookFactory,
    TrafficCSVLoader,
    TrafficDatasetAnalyzer,
)
from src.use_cases.generate_eda_reports import NotebookDocument

pd = pytest.importorskip("pandas")


def test_loader_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"text": ["a"], "category": ["b"]}).to_csv(csv_path, index=False)

    loader = TrafficCSVLoader(csv_path)
    df = loader.load()
    assert list(df.columns) == ["text", "category"]


def test_analyzer_computes_expected_metrics() -> None:
    data = pd.DataFrame(
        {
            "text": ["Accidente en la vía", "Tráfico pesado", "Lluvia intensa"],
            "category": ["accidente", "informativo", "accidente"],
            "created_at": ["2023-01-01", "2023-01-08", "2023-02-01"],
        }
    )
    analyzer = TrafficDatasetAnalyzer()

    metrics = analyzer.compute_metrics(data)
    assert metrics["total_records"] == 3
    assert metrics["category_counts"]["accidente"] == 2
    assert "text_length" in metrics
    assert metrics["top_tokens"]["accidente"] >= 1
    assert metrics["date_range"]["start"].startswith("2023-01-01")

    figures = analyzer.build_figures(data)
    assert "category_distribution" in figures
    assert "text_length_histogram" in figures
    assert "monthly_activity" in figures
    for figure in figures.values():
        assert isinstance(figure, Figure)


def test_repository_persists_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics" / "summary.json"
    figures_dir = tmp_path / "figures"
    notebook_path = tmp_path / "notebooks" / "eda.ipynb"
    figures_dir.mkdir()
    metrics_path.parent.mkdir()
    notebook_path.parent.mkdir()

    repository = FileSystemReportRepository(metrics_path=metrics_path, figures_dir=figures_dir)
    notebook = NotebookDocument(path=notebook_path, content={"cells": []})

    metrics_file = repository.save_metrics({"total_records": 1})
    figure = Figure()
    figure_paths = repository.save_figures({"distribution": figure})
    notebook_file = repository.save_notebook(notebook)

    assert metrics_file.exists()
    saved_figure = figure_paths["distribution"]
    assert saved_figure.exists()
    assert notebook_file.exists()


def test_notebook_factory_creates_markdown_tables(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "text": ["Uno", "Dos"],
            "category": ["informativo", "accidente"],
            "created_at": ["2024-01-01", "2024-01-02"],
        }
    )
    metrics = {
        "total_records": 2,
        "category_counts": {"informativo": 1, "accidente": 1},
        "category_proportions": {"informativo": 0.5, "accidente": 0.5},
        "top_tokens": {"accidente": 1},
        "date_range": {"start": "2024-01-01T00:00:00", "end": "2024-01-02T00:00:00"},
    }
    factory = SimpleNotebookFactory(output_path=tmp_path / "eda.ipynb")

    notebook = factory.build(data, metrics)
    assert notebook.path.name == "eda.ipynb"
    assert "cells" in notebook.content
    assert any("Resumen Ejecutivo" in cell["source"] for cell in notebook.content["cells"])
