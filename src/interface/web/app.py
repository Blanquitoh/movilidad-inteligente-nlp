"""Streamlit interface for the mobility intelligent NLP project."""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, TypedDict, cast
from uuid import uuid4

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

from src.utils.logger import logger

from src.core.entities import DetectedInterest, SentimentPrediction, TrafficEvent, UserProfile
from src.infrastructure.events.priority import TimeSeverityPriorityAssessor
from src.infrastructure.geo.resolver import GeoResolver
from src.infrastructure.nlp.category_rules import KeywordCategoryResolver
from src.infrastructure.nlp.model_builder import TextClassifierPipeline
from src.infrastructure.nlp.sentiment_analysis import SentimentAnalyzer
from src.infrastructure.nlp.severity import KeywordSeverityScorer
from src.infrastructure.recommenders.content_based import ContentBasedRecommender
from src.infrastructure.recommenders.segments import AgeSegmenter
from src.use_cases.analyze_sentiment import AnalyzeSentimentUseCase
from src.use_cases.detect_events import DetectEventsUseCase
from src.use_cases.infer_interests import InferInterestsUseCase
from src.use_cases.recommend_topics import RecommendTopicsUseCase


class PathsConfig(TypedDict, total=False):
    model_artifact: str
    recommendation_data: str
    sentiment_data: str


class TextPreprocessingConfig(TypedDict, total=False):
    stopword_exclusions: list[str]


class ModelConfigEntry(TypedDict, total=False):
    max_features: int
    C: float
    n_neurons: int
    ngram_range: list[int]
    stopword_exclusions: list[str]


class AgeSegmentEntry(TypedDict, total=False):
    label: str
    min: int
    max: int


class DemographicsConfig(TypedDict, total=False):
    age_segments: list[AgeSegmentEntry]


class RecommenderConfig(TypedDict, total=False):
    demographics: DemographicsConfig


class SentimentModelConfig(TypedDict, total=False):
    max_features: int
    C: float
    ngram_range: list[int]
    stopword_exclusions: list[str]
    label_aliases: dict[str, str]
    emotion_aliases: dict[str, str]


class CategoryRuleEntry(TypedDict, total=False):
    category: str
    patterns: list[str]


class CategoryRulesConfig(TypedDict, total=False):
    keyword_overrides: list[CategoryRuleEntry]


class MapsConfig(TypedDict, total=False):
    google_api_key: str


class AppConfig(TypedDict, total=False):
    paths: PathsConfig
    text_preprocessing: TextPreprocessingConfig
    model: ModelConfigEntry
    sentiment_model: SentimentModelConfig
    category_rules: CategoryRulesConfig
    recommender: RecommenderConfig
    maps: MapsConfig
    n_suggestions: int
    top_keywords: int


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
def load_recommender(path: Path, _age_segmenter: AgeSegmenter | None = None) -> ContentBasedRecommender:
    logger.info("Loading recommender data from {}", path)
    try:
        return ContentBasedRecommender.from_csv(
            str(path),
            age_segmenter=_age_segmenter,
        )
    except FileNotFoundError:
        logger.warning("Recommendation data not found; using empty catalog.")
        empty_catalog = pd.DataFrame(columns=["category", "source_text", "age_segment", "gender"])
        return ContentBasedRecommender(
            empty_catalog,
            age_segmenter=_age_segmenter,
        )


@st.cache_resource
def load_sentiment_analyzer(path: Path, params_signature: str = "") -> SentimentAnalyzer:
    logger.info("Loading sentiment analyzer from {}", path)
    params: dict[str, object] = {}
    if params_signature:
        try:
            params = json.loads(params_signature)
        except json.JSONDecodeError:
            logger.warning("Invalid sentiment model configuration; using defaults.")
            params = {}
    try:
        max_features = params.get("max_features")
        if not isinstance(max_features, int):
            max_features = 5000
        C_value = params.get("C")
        if isinstance(C_value, (int, float)):
            C_float = float(C_value)
        else:
            C_float = 1.0
        ngram_range = params.get("ngram_range")
        if not isinstance(ngram_range, list):
            ngram_range = None
        stopword_exclusions = params.get("stopword_exclusions")
        if not isinstance(stopword_exclusions, list):
            stopword_exclusions = None
        label_aliases = params.get("label_aliases")
        if not isinstance(label_aliases, dict):
            label_aliases = None
        emotion_aliases = params.get("emotion_aliases")
        if not isinstance(emotion_aliases, dict):
            emotion_aliases = None
        return SentimentAnalyzer.from_csv(
            str(path),
            max_features=max_features,
            C=C_float,
            ngram_range=ngram_range,
            stopword_exclusions=stopword_exclusions,
            label_aliases=label_aliases,
            emotion_aliases=emotion_aliases,
        )
    except FileNotFoundError:
        logger.warning("Sentiment dataset not found; disabling sentiment predictions.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise sentiment analyzer: {}", exc)
    return SentimentAnalyzer.empty()


def build_age_segmenter(config: AppConfig) -> AgeSegmenter:
    recommender_config = cast(RecommenderConfig, config.get("recommender", {}))
    demographics_config = cast(DemographicsConfig, recommender_config.get("demographics", {}))
    age_segments_config = demographics_config.get("age_segments")

    if isinstance(age_segments_config, list) and age_segments_config:
        return AgeSegmenter.from_config(age_segments_config)

    return AgeSegmenter.default()


def get_google_maps_api_key(config: AppConfig) -> Optional[str]:
    maps_config = cast(MapsConfig, config.get("maps", {}))
    api_key = maps_config.get("google_api_key") if isinstance(maps_config, dict) else None
    if api_key is None:
        return None

    text_key = str(api_key).strip()
    return text_key or None


def build_category_resolver_from_config(config: AppConfig) -> KeywordCategoryResolver:
    category_config = cast(CategoryRulesConfig, config.get("category_rules", {}))
    overrides = category_config.get("keyword_overrides")
    if isinstance(overrides, list) and overrides:
        keyword_map: dict[str, str] = {}
        for entry in overrides:
            category = entry.get("category")
            patterns = entry.get("patterns")
            if not isinstance(category, str) or not category.strip():
                continue
            if not isinstance(patterns, list):
                continue
            for pattern in patterns:
                if isinstance(pattern, str) and pattern.strip():
                    keyword_map[pattern] = category
        if keyword_map:
            return KeywordCategoryResolver(keyword_map=keyword_map)
    return KeywordCategoryResolver.for_obstacles()


def create_use_cases(
    config: AppConfig,
) -> tuple[DetectEventsUseCase, RecommendTopicsUseCase, InferInterestsUseCase, AnalyzeSentimentUseCase]:
    if "paths" not in config:
        raise KeyError("Configuration is missing the 'paths' section.")

    paths: PathsConfig = config["paths"]
    geo_resolver = GeoResolver()
    severity_scorer = KeywordSeverityScorer()
    category_resolver = build_category_resolver_from_config(config)
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
    age_segmenter = build_age_segmenter(config)
    recommender: ContentBasedRecommender = load_recommender(
        recommendation_path,
        _age_segmenter=age_segmenter,
    )

    sentiment_data = paths.get("sentiment_data")
    if sentiment_data is None:
        raise KeyError("Configuration 'paths' is missing the 'sentiment_data' entry.")

    sentiment_path = Path(sentiment_data)
    sentiment_model_config = cast(SentimentModelConfig, config.get("sentiment_model", {}))
    sentiment_signature = (
        json.dumps(sentiment_model_config, sort_keys=True, ensure_ascii=False)
        if sentiment_model_config
        else ""
    )
    sentiment_analyzer = load_sentiment_analyzer(sentiment_path, sentiment_signature)
    sentiment_use_case = AnalyzeSentimentUseCase(sentiment_analyzer)

    detect_use_case = DetectEventsUseCase(
        classifier,
        geo_resolver,
        severity_scorer,
        category_resolver=category_resolver,
        priority_assessor=priority_assessor,
    )
    recommend_use_case = RecommendTopicsUseCase(recommender)
    infer_interests_use_case = InferInterestsUseCase(recommender)
    return detect_use_case, recommend_use_case, infer_interests_use_case, sentiment_use_case


def render_event_table(events: Sequence[TrafficEvent]) -> None:
    if not events:
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


def render_event_map(events: Sequence[TrafficEvent], *, api_key: Optional[str]) -> None:
    points = [
        {
            "lat": event.latitude,
            "lng": event.longitude,
            "category": event.predicted_category or "Evento",
            "text": event.text,
        }
        for event in events
        if event.latitude is not None and event.longitude is not None
    ]

    if not points:
        return

    st.markdown("#### 🗺️ Mapa de ubicaciones detectadas")

    if not api_key:
        st.info(
            "Agrega una clave de Google Maps en el archivo de configuración para visualizar el mapa interactivo."
        )
        return

    element_id = f"event-map-{uuid4().hex}"
    map_html = _build_google_map_html(points, api_key=api_key, element_id=element_id)
    components.html(map_html, height=420)


def _build_google_map_html(
    points: Sequence[dict[str, object]], *, api_key: str, element_id: str
) -> str:
    callback_name = f"initMap_{element_id.replace('-', '_')}"

    sanitized_points: list[dict[str, object]] = []
    for point in points:
        sanitized_point = dict(point)
        text = sanitized_point.get("text")
        if isinstance(text, str):
            sanitized_point["text"] = html.escape(text, quote=False)
        category = sanitized_point.get("category")
        if isinstance(category, str):
            sanitized_point["category"] = html.escape(category, quote=False)
        sanitized_points.append(sanitized_point)

    data_json = json.dumps(sanitized_points, ensure_ascii=False)

    return f"""
<div id="{element_id}" style="height: 420px; border-radius: 12px; overflow: hidden;"></div>
<script>
const mapPoints = {data_json};
function {callback_name}() {{
    const container = document.getElementById('{element_id}');
    if (!container) {{
        return;
    }}
    const map = new google.maps.Map(container, {{
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false
    }});
    const bounds = new google.maps.LatLngBounds();
    mapPoints.forEach((item) => {{
        const position = new google.maps.LatLng(item.lat, item.lng);
        const marker = new google.maps.Marker({{
            position,
            map,
            title: item.category || 'Evento detectado'
        }});
        if (item.text) {{
            const info = new google.maps.InfoWindow({{
                content: `<div style="max-width:260px"><strong>${{item.category || 'Evento detectado'}}</strong><p style="margin:0">${{item.text}}</p></div>`
            }});
            marker.addListener('click', () => info.open({{map, anchor: marker}}));
        }}
        bounds.extend(position);
    }});
    if (!bounds.isEmpty()) {{
        map.fitBounds(bounds);
        if (mapPoints.length === 1) {{
            map.setZoom(13);
        }}
    }}
}}
</script>
<script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback={callback_name}" async defer></script>
"""


def render_detection_section(
    detect_use_case: DetectEventsUseCase,
    maps_api_key: Optional[str],
) -> None:
    st.markdown("### 🛰️ Detección de eventos en tiempo real")
    st.caption("Analiza tweets para identificar incidentes viales y visualiza los resultados en un mapa interactivo.")

    with st.form("detect-form"):
        text_input = st.text_area(
            "Escribe uno o más tweets (uno por línea)",
            height=220,
            placeholder="Ejemplo: Choque en la Av. Central, mucho tráfico...",
        )
        submitted = st.form_submit_button("Analizar tweets", use_container_width=True)

    if not submitted:
        return

    texts = [line.strip() for line in text_input.splitlines() if line.strip()]
    if not texts:
        st.warning("Por favor ingresa al menos un tweet para iniciar el análisis.")
        return

    with st.spinner("Analizando tweets..."):
        events = detect_use_case.execute(texts)

    if not events:
        st.info("No se detectaron eventos en los tweets proporcionados. Intenta con descripciones más detalladas.")
        return

    geolocated = sum(1 for event in events if event.latitude is not None and event.longitude is not None)
    categories = {event.predicted_category for event in events if event.predicted_category}
    col1, col2, col3 = st.columns(3)
    col1.metric("Eventos detectados", len(events))
    col2.metric("Con geolocalización", geolocated)
    col3.metric("Categorías únicas", len(categories))

    render_event_table(events)
    render_event_map(events, api_key=maps_api_key)


def _format_detected_interests(interests: Sequence[DetectedInterest]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for interest in interests:
        keywords = ", ".join(interest.keywords) if interest.keywords else "—"
        rows.append(
            {
                "Interés": interest.name.capitalize(),
                "Coincidencias": interest.score,
                "Palabras clave": keywords,
            }
        )

    return pd.DataFrame(rows, columns=["Interés", "Coincidencias", "Palabras clave"])


def _format_sentiment_predictions(predictions: Sequence[SentimentPrediction]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for prediction in predictions:
        confidence = f"{prediction.probability:.1%}"
        emotion = prediction.emotion.capitalize() if prediction.emotion else "—"
        rows.append(
            {
                "Texto": prediction.text,
                "Sentimiento": prediction.sentiment.capitalize(),
                "Confianza": confidence,
                "Emoción asociada": emotion,
            }
        )

    return pd.DataFrame(rows, columns=["Texto", "Sentimiento", "Confianza", "Emoción asociada"])


def render_recommendation_section(
    recommend_use_case: RecommendTopicsUseCase,
    infer_interests_use_case: InferInterestsUseCase,
    sentiment_use_case: AnalyzeSentimentUseCase,
    *,
    suggestions_limit: int,
    top_keywords_limit: int,
) -> None:
    st.markdown("### 🎯 Recomendaciones personalizadas por perfil")
    st.caption(
        "Pega publicaciones o tweets del usuario para identificar intereses automáticamente y descubrir temas afines."
    )

    with st.form("recommendation-form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            user_identifier = st.text_input("Identificador de usuario", value="usuario_demo")
        with col2:
            omit_age = st.checkbox("Omitir edad", value=False)

        posts_input = st.text_area(
            "Publicaciones del usuario (una por línea)",
            height=220,
            placeholder="Ejemplo: Me encanta salir a correr al amanecer...",
        )

        age_value_input = st.number_input("Edad", min_value=0, max_value=120, value=29, step=1)
        gender_option = st.selectbox(
            "Género",
            options=("Prefiero no decirlo", "Femenino", "Masculino", "Otro"),
            index=0,
        )
        submitted = st.form_submit_button("Generar recomendaciones", use_container_width=True)

    if not submitted:
        return

    posts = [line.strip() for line in posts_input.splitlines() if line.strip()]
    if not posts:
        st.warning("Ingresa al menos una publicación para analizar los intereses del usuario.")
        return

    identifier = user_identifier.strip() or "usuario_demo"
    age_value: Optional[int] = None if omit_age else int(age_value_input)
    gender_value = None if gender_option == "Prefiero no decirlo" else gender_option.lower()

    with st.spinner("Analizando publicaciones..."):
        detected_interests = infer_interests_use_case.execute(posts, max_keywords=top_keywords_limit)
        sentiment_predictions = sentiment_use_case.execute(posts)

    if detected_interests:
        st.markdown("### Intereses detectados automáticamente")
        interests_table = _format_detected_interests(detected_interests)
        st.dataframe(interests_table, use_container_width=True)
    else:
        st.info(
            "No se detectaron intereses en los textos proporcionados. Intenta con publicaciones más descriptivas."
        )

    if sentiment_predictions:
        st.markdown("### Análisis de sentimiento")
        sentiment_table = _format_sentiment_predictions(sentiment_predictions)
        st.dataframe(sentiment_table, use_container_width=True)

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Publicaciones analizadas", len(posts))
    metrics_col2.metric("Intereses detectados", len(detected_interests))
    metrics_col3.metric("Sentimientos analizados", len(sentiment_predictions))

    if not detected_interests:
        return

    profile_interests = [interest.name for interest in detected_interests]
    profile = UserProfile(
        user_id=identifier,
        interests=profile_interests,
        age=age_value,
        gender=gender_value,
    )

    with st.spinner("Generando recomendaciones personalizadas..."):
        topics = recommend_use_case.execute(profile, top_k=max(1, suggestions_limit))

    st.markdown("### Recomendaciones sugeridas")
    if not topics:
        st.info(
            "No se encontraron recomendaciones para los intereses detectados. Intenta con más publicaciones o textos diferentes."
        )
        return

    recommendations_md = "\n".join(f"- 🎯 **{topic}**" for topic in topics)
    st.markdown(recommendations_md)
    st.caption(
        "Las sugerencias se generan a partir del catálogo de contenidos y los intereses identificados automáticamente."
    )


def main() -> None:
    st.set_page_config(page_title="Movilidad Inteligente NLP", layout="wide")
    st.title("🚦 Movilidad Inteligente: Detección de eventos de tráfico")

    config: AppConfig = load_config(Path("configs/config.yaml"))
    (
        detect_use_case,
        recommend_use_case,
        infer_interests_use_case,
        sentiment_use_case,
    ) = create_use_cases(config)
    maps_api_key = get_google_maps_api_key(config)
    suggestions_limit = int(config.get("n_suggestions", 5) or 5)
    top_keywords_limit = int(config.get("top_keywords", 5) or 5)

    detection_tab, recommendation_tab = st.tabs([
        "Detección de eventos",
        "Recomendaciones personalizadas",
    ])

    with detection_tab:
        render_detection_section(detect_use_case, maps_api_key)

    with recommendation_tab:
        render_recommendation_section(
            recommend_use_case,
            infer_interests_use_case,
            sentiment_use_case,
            suggestions_limit=suggestions_limit,
            top_keywords_limit=top_keywords_limit,
        )


if __name__ == "__main__":
    main()
