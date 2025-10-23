"""Command-line entry point to generate analytical artefacts for the dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.bootstrap import bootstrap_project

_PROJECT_ROOT = bootstrap_project()

from src.infrastructure.reports import (  # noqa: E402
    FileSystemReportRepository,
    SimpleNotebookFactory,
    TrafficCSVLoader,
    TrafficDatasetAnalyzer,
)
from src.use_cases.generate_eda_reports import GenerateEdaReportsUseCase  # noqa: E402
from src.utils.logger import logger  # noqa: E402


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera reportes de EDA y artefactos visuales para el dataset procesado"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Ruta del archivo CSV a analizar",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics/dataset_summary.json"),
        help="Ruta destino para el archivo JSON con métricas descriptivas",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directorio donde se guardarán las figuras generadas",
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=Path("notebooks/generated_eda_report.ipynb"),
        help="Ruta del notebook de resumen que se generará",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_path(args.dataset)
    metrics_path = _resolve_path(args.metrics_path)
    figures_dir = _resolve_path(args.figures_dir)
    notebook_path = _resolve_path(args.notebook_path)

    loader = TrafficCSVLoader(dataset_path)
    analyzer = TrafficDatasetAnalyzer()
    notebook_factory = SimpleNotebookFactory(notebook_path)
    repository = FileSystemReportRepository(metrics_path=metrics_path, figures_dir=figures_dir)

    use_case = GenerateEdaReportsUseCase(
        loader=loader,
        analyzer=analyzer,
        notebook_factory=notebook_factory,
        repository=repository,
    )

    reports = use_case.execute()
    logger.info("Métricas guardadas en: {}", reports.metrics_path)
    for name, path in reports.figure_paths.items():
        logger.info("Figura '{}' guardada en {}", name, path)
    logger.info("Notebook generado en: {}", reports.notebook_path)


if __name__ == "__main__":
    main()
