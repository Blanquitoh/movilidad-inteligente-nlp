"""Streamlit interface for the mobility intelligent NLP project."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence, TypedDict, cast

import pandas as pd
import streamlit as st
from loguru import logger
from streamlit.delta_generator import DeltaGenerator

CURRENT_FILE = Path(__file__).resolve()
for candidate in CURRENT_FILE.parents:
    if (candidate / "pyproject.toml").exists():
        project_root = candidate
        break
else:
    project_root = CURRENT_FILE.parents[3]

project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from scripts.bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

from src.core.entities import TrafficEvent, UserProfile
from src.infrastructure.events.priority import TimeSeverityPriorityAssessor
from src.infrastructure.geo.resolver import GeoResolver
from src.infrastructure.nlp.category_rules import KeywordCategoryResolver
from src.infrastructure.nlp.model_builder import TextClassifierPipeline
from src.infrastructure.nlp.severity import KeywordSeverityScorer
from src.infrastructure.recommenders.content_based import ContentBasedRecommender
from src.use_cases.detect_events import DetectEventsUseCase
from src.use_cases.recommend_topics import RecommendTopicsUseCase


class PathsConfig(TypedDict, total=False):
    model_artifact: str
    recommendation_data: str


class AppConfig(TypedDict, total=False):
    paths: PathsConfig


@st.cache_data
def load_config(path: Path) -> AppConfig:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("The configuration file must contain a mapping at the top level.")

    return cast(AppConfig, data)


@st.cache_resource
def load_classifier(path: Path) -> TextClassifierPipeline:
    logger.info("Loading classifier from {}", path)
    return TextClassifierPipeline.load(path)


@st.cache_resource
def load_recommender(path: Path) -> ContentBasedRecommender:
    logger.info("Loading recommender data from {}", path)
    try:
        return ContentBasedRecommender.from_csv(str(path))
    except FileNotFoundError:
        logger.warning("Recommendation data not found; using empty catalog.")
        return ContentBasedRecommender(pd.DataFrame(columns=["category", "content"]))


def create_use_cases(config: AppConfig) -> tuple[DetectEventsUseCase, RecommendTopicsUseCase]:
    if "paths" not in config:
        raise KeyError("Configuration is missing the 'paths' section.")

    paths: PathsConfig = config["paths"]
    geo_resolver = GeoResolver()
    severity_scorer = KeywordSeverityScorer()
    category_resolver = KeywordCategoryResolver.for_obstacles()
    priority_assessor = TimeSeverityPriorityAssessor()

    model_artifact = paths.get("model_artifact")
    if model_artifact is None:
        raise KeyError("Configuration 'paths' is missing the 'model_artifact' entry.")

    model_path = Path(model_artifact)
    try:
        classifier: TextClassifierPipeline = load_classifier(model_path)
    except FileNotFoundError:
        classifier = load_classifier(model_path.with_suffix(""))

    recommendation_data = paths.get("recommendation_data")
    if recommendation_data is None:
        raise KeyError("Configuration 'paths' is missing the 'recommendation_data' entry.")

    recommendation_path = Path(recommendation_data)
    recommender: ContentBasedRecommender = load_recommender(recommendation_path)

    detect_use_case = DetectEventsUseCase(
        classifier,
        geo_resolver,
        severity_scorer,
        category_resolver=category_resolver,
        priority_assessor=priority_assessor,
    )
    recommend_use_case = RecommendTopicsUseCase(recommender)
    return detect_use_case, recommend_use_case


def render_event_table(events: Sequence[TrafficEvent]) -> None:
    if not events:
        st.info("Ingresa un tweet para detectar eventos.")
        return

    sorted_events = sorted(
        events,
        key=lambda event: (
            event.priority_score if event.priority_score is not None else -1.0,
            event.created_at or datetime.min,
        ),
        reverse=True,
    )

    rows: list[dict[str, str | float | None | datetime]] = []
    for event in sorted_events:
        map_link = ""
        if event.latitude is not None and event.longitude is not None:
            map_link = f"https://www.google.com/maps?q={event.latitude},{event.longitude}"
        row: dict[str, str | float | None | datetime] = {
            "Texto": event.text,
            "Categoría": event.predicted_category,
            "Ubicación": event.location_name or "N/D",
            "Lat": event.latitude,
            "Lon": event.longitude,
            "Severidad": event.severity or "N/A",
            "Prioridad": event.priority or "N/A",
            "Puntaje prioridad": event.priority_score,
            "Creado": event.created_at,
            "Mapa": map_link,
        }
        rows.append(row)

    df: pd.DataFrame = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        column_config={
            "Creado": st.column_config.DatetimeColumn("Creado"),
            "Puntaje prioridad": st.column_config.NumberColumn(
                "Puntaje prioridad", format="%.2f"
            ),
            "Mapa": st.column_config.LinkColumn(
                "Mapa", display_text="Abrir en Google Maps"
            )
        },
    )

    severe_events = [
        event
        for event in events
        if event.predicted_category == "accidente" and event.severity == "alta"
    ]
    if severe_events:
        st.error("⚠️ Accidentes de alta severidad detectados. Toma precauciones.")


def main() -> None:
    st.set_page_config(page_title="Movilidad Inteligente NLP", layout="wide")
    st.title("🚦 Movilidad Inteligente: Detección de eventos de tráfico")

    config: AppConfig = load_config(Path("configs/config.yaml"))
    detect_use_case, recommend_use_case = create_use_cases(config)

    with st.sidebar:
        st.header("Recomendaciones personalizadas")
        synthetic_user = UserProfile(user_id="demo", interests=["deporte", "salud", "cine"])
        recommendations = recommend_use_case.execute(synthetic_user, top_k=5)
        st.write("Intereses sugeridos:")
        for topic in recommendations:
            st.markdown(f"- {topic}")

    st.subheader("Detección de eventos")
    text_input = st.text_area("Escribe uno o más tweets (uno por línea)", height=200)
    if st.button("Detectar evento"):
        texts = [line.strip() for line in text_input.splitlines() if line.strip()]
        if not texts:
            st.warning("Por favor ingresa al menos un tweet.")
        else:
            events = detect_use_case.execute(texts)
            render_event_table(events)


if __name__ == "__main__":
    main()
