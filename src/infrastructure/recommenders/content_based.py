"""Simple content-based recommender using keyword overlap."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.core.entities import DetectedInterest, UserProfile
from src.infrastructure.recommenders.segments import AgeSegmenter
from src.utils.logger import logger
from src.utils.text_cleaning import ensure_stopwords


_CANDIDATE_TEXT_COLUMNS = (
    "Post Text",
    "Hashtags",
    "User Bio",
    "User Description 1",
    "User Description 2",
)

_PRIORITY_TEXT_COLUMNS = (
    "Hashtags",
    "User Bio",
    "User Description 1",
    "User Description 2",
)

_INTEREST_TOKEN_HINTS: set[str] = {
    "arte",
    "bienestar",
    "ciencia",
    "cine",
    "cultura",
    "deporte",
    "deportes",
    "educacion",
    "empleo",
    "finanzas",
    "gastronomia",
    "historia",
    "innovacion",
    "juegos",
    "movilidad",
    "musica",
    "negocios",
    "salud",
    "seguridad",
    "tecnologia",
    "tecnologias",
    "transporte",
    "turismo",
    "viajes",
}

_MAX_SEGMENT_DISTANCE_FOR_FALLBACK = 1


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalise_token(token: str) -> str:
    token = token.strip().lower()
    if not token:
        return ""
    return _strip_accents(token)


def _tokenise(text: str) -> list[str]:
    processed = re.sub(r"(?<=[a-zA-Z])(?=[A-Z])", " ", str(text))
    processed = processed.replace("#", " ").replace("_", " ")
    lowered = processed.lower()
    return re.findall(r"\b[\wáéíóúüñ]+\b", lowered, flags=re.UNICODE)


def _build_stopwords() -> set[str]:
    spanish = ensure_stopwords()
    english = ENGLISH_STOP_WORDS
    combined = {_normalise_token(token) for token in spanish.union(english)}
    combined.discard("")
    return combined


_STOPWORDS = _build_stopwords()


@dataclass(frozen=True)
class _RowTokens:
    priority: set[str]
    all: set[str]
    priority_weights: dict[str, int]


_COLUMN_WEIGHTS: dict[str, int] = {
    "User Description 1": 3,
    "User Description 2": 3,
    "User Bio": 3,
    "Hashtags": 2,
}


def _extract_row_tokens(row: pd.Series) -> _RowTokens:
    priority_tokens: set[str] = set()
    priority_weights: dict[str, int] = {}
    all_tokens: set[str] = set()
    for column in _CANDIDATE_TEXT_COLUMNS:
        if column not in row:
            continue
        value = row.get(column)
        if not value or not isinstance(value, str):
            continue
        weight = _COLUMN_WEIGHTS.get(column, 1)
        for token in _tokenise(value):
            normalised = _normalise_token(token)
            if len(normalised) <= 2:
                continue
            if normalised in _STOPWORDS:
                continue
            if not normalised.isalpha():
                continue
            all_tokens.add(normalised)
            if column in _PRIORITY_TEXT_COLUMNS:
                priority_tokens.add(normalised)
                existing = priority_weights.get(normalised, 0)
                if weight > existing:
                    priority_weights[normalised] = weight
    return _RowTokens(priority=priority_tokens, all=all_tokens, priority_weights=priority_weights)


def _build_document_frequencies(rows_tokens: Sequence[_RowTokens]) -> Counter[str]:
    doc_freq: Counter[str] = Counter()
    for tokens in rows_tokens:
        for token in set(tokens.all):
            doc_freq[token] += 1
    return doc_freq


def _rank_tokens(
    tokens: Iterable[str],
    doc_freq: Counter[str],
    total_docs: int,
    *,
    weights: Mapping[str, int] | None = None,
) -> list[str]:
    unique_tokens = set(tokens)
    if not unique_tokens:
        return []

    return sorted(
        unique_tokens,
        key=lambda token: (
            -(weights or {}).get(token, 0),
            -(token in _INTEREST_TOKEN_HINTS),
            -doc_freq[token],
            len(token),
            token,
        ),
    )


def _select_candidate_tokens(
    row_tokens: _RowTokens,
    doc_freq: Counter[str],
    total_docs: int,
    max_tokens: int = 3,
) -> list[str]:
    selected: list[str] = []

    def extend(
        tokens: Iterable[str],
        *,
        weights: Mapping[str, int] | None = None,
        require_hint: bool,
    ) -> None:
        ranked = _rank_tokens(tokens, doc_freq, total_docs, weights=weights)
        for token in ranked:
            if token in selected:
                continue
            if require_hint and token not in _INTEREST_TOKEN_HINTS:
                continue
            selected.append(token)
            if len(selected) >= max_tokens:
                return

    extend(
        row_tokens.priority,
        weights=row_tokens.priority_weights,
        require_hint=True,
    )

    if len(selected) < max_tokens:
        extend(row_tokens.all, weights=None, require_hint=True)

    if not selected:
        extend(
            row_tokens.priority,
            weights=row_tokens.priority_weights,
            require_hint=False,
        )

    if not selected:
        extend(row_tokens.all, weights=None, require_hint=False)

    return selected


def _coerce_age(value: Any) -> Optional[int]:
    """Convert raw age values into integers when possible."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        if value != value:  # NaN check
            return None
        return int(value)

    text = str(value).strip()
    if not text or not any(char.isdigit() for char in text):
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass(frozen=True)
class _InterestLexicon:
    """Lexical rules that map free text into high level interests."""

    vocabulary: dict[str, set[str]]

    @classmethod
    def default(cls) -> "_InterestLexicon":
        return cls(vocabulary={})

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "_InterestLexicon":
        if dataframe.empty:
            logger.warning("Recommendation dataset is empty; returning empty lexicon.")
            return cls.default()

        rows_tokens: list[_RowTokens] = []
        for _, row in dataframe.iterrows():
            tokens = _extract_row_tokens(row)
            rows_tokens.append(tokens)

        doc_freq = _build_document_frequencies(rows_tokens)
        total_docs = len(rows_tokens)

        vocabulary: dict[str, set[str]] = defaultdict(set)
        for row_tokens in rows_tokens:
            labels = _select_candidate_tokens(row_tokens, doc_freq, total_docs)
            if not labels:
                continue
            for label in labels:
                vocabulary[label].update(row_tokens.all)

        if not vocabulary:
            logger.warning("Failed to derive lexicon from dataset; returning empty vocabulary.")
            return cls.default()

        return cls(vocabulary=dict(vocabulary))

    def detect(self, text: str) -> set[str]:
        if not text:
            return set()

        tokens = {_normalise_token(token) for token in _tokenise(text)}
        tokens.discard("")
        matches: set[str] = set()
        for category, keywords in self.vocabulary.items():
            if tokens.intersection(keywords):
                matches.add(category)
        return matches


@dataclass
class KeywordInterestDetector:
    """Detect high-level interests based on keyword occurrences."""

    lexicon: _InterestLexicon

    def detect(self, texts: Sequence[str], max_keywords: int = 5) -> list[DetectedInterest]:
        aggregated = Counter()
        keyword_counts: dict[str, Counter[str]] = {}

        for raw_text in texts:
            if not raw_text:
                continue

            tokens = {_normalise_token(token) for token in _tokenise(raw_text)}
            valid_tokens = {token for token in tokens if token}

            for category, keywords in self.lexicon.vocabulary.items():
                intersection = valid_tokens.intersection(keywords)
                if not intersection:
                    continue
                aggregated[category] += 1
                counter = keyword_counts.setdefault(category, Counter())
                for keyword in intersection:
                    counter[keyword] += 1

        detections: list[DetectedInterest] = []
        for category, count in aggregated.most_common():
            keywords_counter = keyword_counts.get(category, Counter())
            top_keywords = tuple(
                keyword for keyword, _ in keywords_counter.most_common(max(0, int(max_keywords)))
            )
            detections.append(DetectedInterest(name=category, score=count, keywords=top_keywords))

        return detections


class ContentBasedRecommender:
    """Recommend topics that overlap with the user's interests."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        lexicon: _InterestLexicon | None = None,
        age_segmenter: AgeSegmenter | None = None,
    ) -> None:
        catalog = catalog.copy()
        self._lexicon = lexicon or _InterestLexicon.from_dataframe(catalog)
        self._allow_segment_fallback = age_segmenter is None
        self._age_segmenter = age_segmenter or AgeSegmenter.default()
        self._segment_positions = {
            label: index for index, label in enumerate(self._age_segmenter.labels())
        }
        self._interest_detector = KeywordInterestDetector(self._lexicon)
        self.catalog = self._normalise_catalog(
            catalog,
            self._lexicon,
            age_segmenter=self._age_segmenter,
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        lexicon: _InterestLexicon | None = None,
        age_segmenter: AgeSegmenter | None = None,
    ) -> "ContentBasedRecommender":
        logger.info("Loading recommendation data from {}", path)
        catalog = pd.read_csv(path)
        derived_lexicon = lexicon or _InterestLexicon.from_dataframe(catalog)
        return cls(catalog, lexicon=derived_lexicon, age_segmenter=age_segmenter)

    @staticmethod
    def _normalise_catalog(
        catalog: pd.DataFrame,
        lexicon: _InterestLexicon,
        age_segmenter: AgeSegmenter | None = None,
    ) -> pd.DataFrame:
        if catalog.empty:
            return pd.DataFrame(columns=["category", "source_text", "age_segment", "gender"])

        segmenter = age_segmenter or AgeSegmenter.default()

        text_columns = [
            column
            for column in (
                "category",
                "categories",
                "Hashtags",
                "Post Text",
                "User Description 1",
                "User Description 2",
                "User Bio",
            )
            if column in catalog.columns
        ]

        records: list[dict[str, Optional[str]]] = []
        for _, row in catalog.iterrows():
            values = [
                str(row.get(column, ""))
                for column in text_columns
                if pd.notna(row.get(column, "")) and str(row.get(column, "")).strip()
            ]
            text = " ".join(values)
            detected = lexicon.detect(text)
            age_value = row.get("User Age") or row.get("user_age")
            age_segment = segmenter.segment(_coerce_age(age_value))
            gender_value = row.get("User Gender") or row.get("user_gender")
            gender = str(gender_value).strip().lower() if pd.notna(gender_value) and str(gender_value).strip() else None
            for category in detected:
                records.append(
                    {
                        "category": category,
                        "source_text": text,
                        "age_segment": age_segment,
                        "gender": gender,
                    }
                )

        if not records:
            logger.warning("No interest categories detected in recommendation dataset")
            return pd.DataFrame(columns=["category", "source_text", "age_segment", "gender"])

        normalised = pd.DataFrame(records).drop_duplicates()
        logger.debug("Prepared {} recommendation records", len(normalised))
        return normalised

    def recommend(self, user: UserProfile, top_k: int = 5) -> list[str]:
        if self.catalog.empty:
            return []

        logger.debug(
            "Recommending topics for user interests: {} with age {}",
            user.interests,
            user.age,
        )

        primary_catalog = self.catalog
        fallback_catalog: pd.DataFrame | None = None
        user_segment = self._age_segmenter.segment(user.age)

        if user_segment:
            segment_index = self._segment_positions.get(user_segment)
            segmented = self.catalog[self.catalog["age_segment"] == user_segment]
            if not segmented.empty:
                primary_catalog = segmented
                if self._allow_segment_fallback:
                    allowed_segments: set[str] = set()
                    if segment_index is not None:
                        for segment_label, index in self._segment_positions.items():
                            if segment_label == user_segment:
                                continue
                            if abs(index - segment_index) <= _MAX_SEGMENT_DISTANCE_FOR_FALLBACK:
                                allowed_segments.add(segment_label)
                    fallback_mask = self.catalog["age_segment"].isin(allowed_segments)
                    fallback_mask |= self.catalog["age_segment"].isna()
                    if fallback_mask.any():
                        fallback_catalog = self.catalog[fallback_mask]
                        if not fallback_catalog.empty:
                            fallback_catalog = fallback_catalog[~fallback_catalog.index.isin(primary_catalog.index)]
                            if fallback_catalog.empty:
                                fallback_catalog = None
            else:
                primary_catalog = self.catalog

        preference_counts = Counter(interest.lower() for interest in user.interests)

        def score_row(row: pd.Series) -> float:
            category = str(row.get("category", "")).strip().lower()
            return preference_counts.get(category, 0)

        def compute_recommendations(data: pd.DataFrame) -> list[str]:
            if data.empty:
                return []

            scored = data.assign(score=data.apply(score_row, axis=1))
            scored = scored[scored["score"] > 0]
            if not scored.empty:
                top = (
                    scored.sort_values(["score", "category"], ascending=[False, True])
                    .drop_duplicates(subset=["category"])
                    .astype({"category": str})
                )
                return top["category"].tolist()

            categories = (
                data["category"].dropna().astype(str).str.strip()
            )
            return [category for category in sorted(set(categories)) if category]

        recommendations: list[str] = []
        for category in compute_recommendations(primary_catalog):
            if category and category not in recommendations:
                recommendations.append(category)
            if len(recommendations) >= top_k:
                return recommendations[:top_k]

        if fallback_catalog is not None:
            for category in compute_recommendations(fallback_catalog):
                if category and category not in recommendations:
                    recommendations.append(category)
                if len(recommendations) >= top_k:
                    break

        return recommendations[:top_k]

    def available_topics(self) -> list[str]:
        if self.catalog.empty:
            return []

        categories = (
            self.catalog["category"].dropna().astype(str).str.strip()
        )
        return sorted({category for category in categories if category})

    def detect_interests(self, texts: Sequence[str], max_keywords: int = 5) -> list[DetectedInterest]:
        return self._interest_detector.detect(texts, max_keywords=max_keywords)


__all__ = ["ContentBasedRecommender", "KeywordInterestDetector"]
