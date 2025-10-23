"""Tests for the GenerateEdaReportsUseCase orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure

from src.use_cases.generate_eda_reports import (
    GenerateEdaReportsUseCase,
    GeneratedReports,
    NotebookDocument,
)

pd = pytest.importorskip("pandas")


class StubLoader:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.called = False

    def load(self) -> pd.DataFrame:
        self.called = True
        return self.data


class StubAnalyzer:
    def __init__(self, metrics: dict, figures: dict[str, Figure]) -> None:
        self._metrics = metrics
        self._figures = figures
        self.calls: list[str] = []

    def compute_metrics(self, data: pd.DataFrame) -> dict:
        self.calls.append("metrics")
        return self._metrics

    def build_figures(self, data: pd.DataFrame) -> dict[str, Figure]:
        self.calls.append("figures")
        return self._figures


@dataclass
class StubNotebookFactory:
    document: NotebookDocument

    def build(self, data: pd.DataFrame, metrics: dict) -> NotebookDocument:
        return self.document


class StubRepository:
    def __init__(self, base: Path) -> None:
        self.metrics_path = base / "metrics.json"
        self.figure_dir = base / "figures"
        self.figure_dir.mkdir()
        self.notebook_path = base / "report.ipynb"
        self.saved_figures: dict[str, Path] = {}

    def save_metrics(self, metrics: dict) -> Path:
        self.metrics_path.write_text("{}", encoding="utf-8")
        return self.metrics_path

    def save_figures(self, figures: dict[str, Figure]) -> dict[str, Path]:
        for name, figure in figures.items():
            path = self.figure_dir / f"{name}.png"
            figure.savefig(path)
            self.saved_figures[name] = path
        return self.saved_figures

    def save_notebook(self, notebook: NotebookDocument) -> Path:
        notebook.path.write_text("{}", encoding="utf-8")
        return notebook.path


def test_use_case_generates_reports(tmp_path: Path) -> None:
    data = pd.DataFrame({"text": ["foo", "bar"], "category": ["a", "b"]})
    loader = StubLoader(data)
    figures = {"hist": Figure()}
    analyzer = StubAnalyzer(metrics={"total_records": 2}, figures=figures)
    document = NotebookDocument(path=tmp_path / "report.ipynb", content={"cells": []})
    notebook_factory = StubNotebookFactory(document=document)
    repository = StubRepository(tmp_path)

    use_case = GenerateEdaReportsUseCase(loader, analyzer, notebook_factory, repository)

    result = use_case.execute()

    assert isinstance(result, GeneratedReports)
    assert result.metrics_path == repository.metrics_path
    assert result.figure_paths == repository.saved_figures
    assert result.notebook_path == document.path
    assert loader.called
    assert analyzer.calls == ["metrics", "figures"]


def test_use_case_fails_with_empty_dataset(tmp_path: Path) -> None:
    loader = StubLoader(pd.DataFrame(columns=["text", "category"]))
    analyzer = StubAnalyzer(metrics={}, figures={})
    document = NotebookDocument(path=tmp_path / "report.ipynb", content={})
    notebook_factory = StubNotebookFactory(document=document)
    repository = StubRepository(tmp_path)
    use_case = GenerateEdaReportsUseCase(loader, analyzer, notebook_factory, repository)

    try:
        use_case.execute()
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected the use case to raise ValueError for empty dataset")
