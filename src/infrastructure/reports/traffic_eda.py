"""Concrete implementations for generating EDA artefacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.use_cases.generate_eda_reports import NotebookDocument


class TrafficCSVLoader:
    """Load traffic-related datasets stored as CSV files."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = Path(csv_path)

    def load(self) -> pd.DataFrame:
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._csv_path}")

        data = pd.read_csv(self._csv_path)
        required_columns = {"text", "category"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(sorted(missing))
            )
        return data


class TrafficDatasetAnalyzer:
    """Compute descriptive statistics and illustrative figures."""

    def __init__(
        self,
        text_column: str = "text",
        category_column: str = "category",
        datetime_column: str = "created_at",
    ) -> None:
        self._text_column = text_column
        self._category_column = category_column
        self._datetime_column = datetime_column

    def compute_metrics(self, data: pd.DataFrame) -> Mapping[str, Any]:
        frame = data.copy()

        total_records = int(len(frame))
        category_counts_series = (
            frame[self._category_column]
            .astype(str)
            .str.lower()
            .value_counts()
            .sort_values(ascending=False)
        )
        category_counts = {
            str(category): int(count) for category, count in category_counts_series.items()
        }
        category_proportions = {
            category: round(count / total_records, 4)
            for category, count in category_counts.items()
            if total_records > 0
        }

        text_lengths = frame[self._text_column].astype(str).str.len()
        text_length_stats = {
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "mean": round(float(text_lengths.mean()), 2),
            "median": round(float(text_lengths.median()), 2),
            "std": round(float(text_lengths.std(ddof=0)), 2),
        }

        date_range = self._extract_date_range(frame)
        top_tokens = self._compute_top_tokens(frame)

        metrics: dict[str, Any] = {
            "total_records": total_records,
            "category_counts": category_counts,
            "category_proportions": category_proportions,
            "text_length": text_length_stats,
            "top_tokens": top_tokens,
        }
        if date_range is not None:
            metrics["date_range"] = date_range
        return metrics

    def _extract_date_range(self, data: pd.DataFrame) -> Mapping[str, str] | None:
        if self._datetime_column not in data.columns:
            return None

        parsed = pd.to_datetime(data[self._datetime_column], errors="coerce")
        parsed = parsed.dropna()
        if parsed.empty:
            return None

        start = parsed.min()
        end = parsed.max()
        return {
            "start": self._format_timestamp(start),
            "end": self._format_timestamp(end),
        }

    def _compute_top_tokens(self, data: pd.DataFrame, limit: int = 15) -> Mapping[str, int]:
        tokens = (
            data[self._text_column]
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.split()
        )
        exploded = tokens.explode()
        filtered = exploded[exploded.notna() & (exploded.str.len() > 2)]
        top_series = filtered.value_counts().head(limit)
        return {str(word): int(count) for word, count in top_series.items()}

    def _format_timestamp(self, value: pd.Timestamp) -> str:
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def build_figures(self, data: pd.DataFrame) -> Mapping[str, Figure]:
        figures: dict[str, Figure] = {}

        counts = (
            data[self._category_column]
            .astype(str)
            .str.lower()
            .value_counts()
            .sort_values(ascending=False)
        )
        if not counts.empty:
            fig_counts, ax_counts = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax_counts, color="#1f77b4")
            ax_counts.set_title("Distribución por categoría")
            ax_counts.set_xlabel("Categoría")
            ax_counts.set_ylabel("Número de registros")
            fig_counts.tight_layout()
            figures["category_distribution"] = fig_counts

        lengths = data[self._text_column].astype(str).str.len()
        if not lengths.empty:
            fig_lengths, ax_lengths = plt.subplots(figsize=(8, 4))
            ax_lengths.hist(lengths, bins=30, color="#ff7f0e", edgecolor="black")
            ax_lengths.set_title("Distribución de longitud de texto")
            ax_lengths.set_xlabel("Caracteres por tweet")
            ax_lengths.set_ylabel("Frecuencia")
            fig_lengths.tight_layout()
            figures["text_length_histogram"] = fig_lengths

        if self._datetime_column in data.columns:
            dates = pd.to_datetime(data[self._datetime_column], errors="coerce").dropna()
            if not dates.empty:
                monthly = dates.dt.to_period("M").value_counts().sort_index()
                if not monthly.empty:
                    fig_timeline, ax_timeline = plt.subplots(figsize=(8, 4))
                    ax_timeline.plot(
                        monthly.index.to_timestamp(),
                        monthly.values,
                        marker="o",
                        color="#2ca02c",
                    )
                    ax_timeline.set_title("Eventos por mes")
                    ax_timeline.set_xlabel("Mes")
                    ax_timeline.set_ylabel("Número de registros")
                    ax_timeline.tick_params(axis="x", rotation=45)
                    fig_timeline.tight_layout()
                    figures["monthly_activity"] = fig_timeline

        return figures


@dataclass(frozen=True)
class _NotebookContent:
    title: str
    narrative: list[str]
    category_table: list[str]
    token_table: list[str]
    sample_table: list[str]


class SimpleNotebookFactory:
    """Render dataset highlights into a lightweight Jupyter notebook."""

    def __init__(self, output_path: Path, sample_columns: Iterable[str] | None = None) -> None:
        self._output_path = Path(output_path)
        self._sample_columns = tuple(sample_columns) if sample_columns is not None else (
            "text",
            "category",
            "created_at",
            "location",
        )

    def build(self, data: pd.DataFrame, metrics: Mapping[str, Any]) -> NotebookDocument:
        content = self._prepare_content(data, metrics)
        notebook_dict = self._render_notebook(content)
        return NotebookDocument(path=self._output_path, content=notebook_dict)

    def _prepare_content(self, data: pd.DataFrame, metrics: Mapping[str, Any]) -> _NotebookContent:
        total_records = metrics.get("total_records", len(data))
        date_range = metrics.get("date_range")
        narrative_lines = [
            f"- Registros analizados: **{total_records}**",
            f"- Columnas disponibles: **{', '.join(data.columns)}**",
        ]
        if date_range:
            narrative_lines.append(
                f"- Rango temporal: **{date_range['start']}** a **{date_range['end']}**"
            )

        category_table = self._build_markdown_table(
            headers=("Categoría", "Frecuencia", "Proporción"),
            rows=[
                (
                    category,
                    str(metrics["category_counts"].get(category, 0)),
                    f"{metrics['category_proportions'].get(category, 0.0):.2%}",
                )
                for category in metrics.get("category_counts", {})
            ],
        )

        token_table = self._build_markdown_table(
            headers=("Token", "Frecuencia"),
            rows=[(token, str(count)) for token, count in metrics.get("top_tokens", {}).items()],
        )

        sample_columns = [col for col in self._sample_columns if col in data.columns]
        if not sample_columns:
            sample_columns = list(data.columns[:5])
        sample = data.loc[:, sample_columns].head(5).copy()
        sample_table = self._build_markdown_table(
            headers=tuple(sample.columns),
            rows=[tuple(str(value) for value in row) for row in sample.itertuples(index=False)],
        )

        return _NotebookContent(
            title="Exploración de Datos del Proyecto",
            narrative=narrative_lines,
            category_table=category_table,
            token_table=token_table,
            sample_table=sample_table,
        )

    def _render_notebook(self, content: _NotebookContent) -> Mapping[str, Any]:
        cells = [
            self._markdown_cell(f"# {content.title}\n"),
            self._markdown_cell("## Resumen Ejecutivo\n" + "\n".join(content.narrative)),
            self._markdown_cell("## Distribución por Categoría\n" + "\n".join(content.category_table)),
            self._markdown_cell("## Tokens Más Frecuentes\n" + "\n".join(content.token_table)),
            self._markdown_cell("## Muestras Iniciales\n" + "\n".join(content.sample_table)),
        ]

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return notebook

    def _markdown_cell(self, content: str) -> Mapping[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": content}

    def _build_markdown_table(
        self, headers: Iterable[str], rows: Iterable[Iterable[str]]
    ) -> list[str]:
        headers_tuple = tuple(headers)
        header_row = "| " + " | ".join(headers_tuple) + " |"
        separator = "| " + " | ".join(["---"] * len(headers_tuple)) + " |"
        body_rows = ["| " + " | ".join(row) + " |" for row in rows]
        return [header_row, separator, *body_rows]


class FileSystemReportRepository:
    """Persist metrics, figures and notebooks to the local filesystem."""

    def __init__(self, metrics_path: Path, figures_dir: Path) -> None:
        self._metrics_path = Path(metrics_path)
        self._figures_dir = Path(figures_dir)

    def save_metrics(self, metrics: Mapping[str, Any]) -> Path:
        self._ensure_parent_exists(self._metrics_path)
        serialisable = json.dumps(metrics, ensure_ascii=False, indent=2)
        self._metrics_path.write_text(serialisable, encoding="utf-8")
        return self._metrics_path

    def save_figures(self, figures: Mapping[str, Figure]) -> Mapping[str, Path]:
        if not self._figures_dir.exists():
            raise FileNotFoundError(
                f"Expected figures directory to exist: {self._figures_dir}"
            )
        saved_paths: dict[str, Path] = {}
        for name, figure in figures.items():
            destination = self._figures_dir / f"{name}.png"
            figure.savefig(destination, dpi=150, bbox_inches="tight")
            plt.close(figure)
            saved_paths[name] = destination
        return saved_paths

    def save_notebook(self, notebook: NotebookDocument) -> Path:
        self._ensure_parent_exists(notebook.path)
        with notebook.path.open("w", encoding="utf-8") as file:
            json.dump(notebook.content, file, ensure_ascii=False, indent=2)
        return notebook.path

    def _ensure_parent_exists(self, path: Path) -> None:
        parent = path.parent
        if not parent.exists():
            raise FileNotFoundError(f"Expected directory to exist: {parent}")


__all__ = [
    "FileSystemReportRepository",
    "SimpleNotebookFactory",
    "TrafficCSVLoader",
    "TrafficDatasetAnalyzer",
]
