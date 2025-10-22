"""Streamlit interface for the mobility intelligent NLP project."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.entities import TrafficEvent, UserProfile
from src.infrastructure.geo.resolver import GeoResolver
from src.infrastructure.nlp.model_builder import TextClassifierPipeline
from src.infrastructure.nlp.severity import KeywordSeverityScorer
from src.infrastructure.recommenders.content_based import ContentBasedRecommender
from src.use_cases.detect_events import DetectEventsUseCase
from src.use_cases.recommend_topics import RecommendTopicsUseCase


@st.cache_data
def load_config(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def create_use_cases(config: dict) -> tuple[DetectEventsUseCase, RecommendTopicsUseCase]:
    geo_resolver = GeoResolver()
    severity_scorer = KeywordSeverityScorer()

    model_path = Path(config["paths"]["model_artifact"])
    try:
        classifier = load_classifier(model_path)
    except FileNotFoundError:
        classifier = load_classifier(model_path.with_suffix(""))

    recommendation_path = Path(config["paths"]["recommendation_data"])
    recommender = load_recommender(recommendation_path)

    detect_use_case = DetectEventsUseCase(classifier, geo_resolver, severity_scorer)
    recommend_use_case = RecommendTopicsUseCase(recommender)
    return detect_use_case, recommend_use_case


def render_event_table(events: List[TrafficEvent]) -> None:
    if not events:
        st.info("Ingresa un tweet para detectar eventos.")
        return

    rows = []
    for event in events:
        row = {
            "Texto": event.text,
            "Categoría": event.predicted_category,
            "Ubicación": event.location_name or "N/D",
            "Lat": event.latitude,
            "Lon": event.longitude,
            "Severidad": event.severity or "N/A",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    severe_events = [event for event in events if event.predicted_category == "accidente" and event.severity == "alta"]
    if severe_events:
        st.error("⚠️ Accidentes de alta severidad detectados. Toma precauciones.")


def main() -> None:
    st.set_page_config(page_title="Movilidad Inteligente NLP", layout="wide")
    st.title("🚦 Movilidad Inteligente: Detección de eventos de tráfico")

    config = load_config(Path("configs/config.yaml"))
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
