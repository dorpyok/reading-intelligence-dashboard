from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_K = 6
DEFAULT_MAX_K = 18
DEFAULT_N_INIT = 20

POSITIVE_RATING_MIN = 4
NEGATIVE_RATING_MAX = 2

OBSERVED_STATUSES = {
    "read",
    "did_not_finish",
    "currently_reading",
}

GENERIC_ATTRIBUTE_TERMS = {
    "book",
    "books",
    "fiction",
    "literature",
    "literary",
    "novel",
    "novels",
    "story",
    "stories",
    "reading",
    "readers",
    "reader",
    "author",
    "authors",
    "writing",
    "written",
    "publication",
    "publications",
    "general",
    "subjects",
    "subject",
    "new",
}

METADATA_ATTRIBUTE_TERMS = {
    "goodreads",
    "isbn",
    "rating",
    "ratings",
    "review",
    "reviews",
    "http",
    "https",
    "www",
    "amp",
}

ENTITY_ATTRIBUTE_TERMS = {
    "new york",
    "new york city",
    "united states",
    "usa",
    "england",
    "america",
}


@dataclass(frozen=True)
class ClusterConfig:
    min_k: int = DEFAULT_MIN_K
    max_k: int = DEFAULT_MAX_K
    n_init: int = DEFAULT_N_INIT
    random_state: int = DEFAULT_RANDOM_STATE


def parse_list_value(value: object) -> list[str]:
    """Safely parse list-like values stored in enriched CSVs."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [text]


def normalize_attribute(value: object) -> str:
    """Normalize a multi-label book attribute without inventing a taxonomy."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[/_]+", " ", text)
    text = re.sub(r"[^a-z0-9'\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if text in GENERIC_ATTRIBUTE_TERMS:
        return ""

    if text in METADATA_ATTRIBUTE_TERMS:
        return ""

    if text in ENTITY_ATTRIBUTE_TERMS:
        return ""

    return text


def extract_book_attributes(row: pd.Series) -> list[str]:
    """
    Extract multi-label attributes from Open Library subjects.

    Attributes are evidence-backed labels. A book may have any number
    of attributes; there is no one-label-per-book assumption.
    """
    values = parse_list_value(row.get("subjects"))
    attributes = {
        normalized
        for value in values
        if (normalized := normalize_attribute(value))
    }
    return sorted(attributes)


def add_book_attributes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["attributes"] = result.apply(extract_book_attributes, axis=1)
    result["attribute_count"] = result["attributes"].str.len()
    return result


def attributes_to_text(attributes: list[str]) -> str:
    return " ".join(attributes)


def _book_identity(row: pd.Series) -> str:
    source_id = str(row.get("source_book_id", "")).strip()
    if source_id:
        return f"id:{source_id}"

    title = str(row.get("title", "")).strip().lower()
    author = str(row.get("author", "")).strip().lower()
    return f"title_author:{title}|{author}"


def deduplicate_books(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate a pooled multi-reader corpus for shared semantic clusters.

    The same Goodreads source book ID represents the same source-level
    book entity for this V1 experiment. Cross-source work/edition identity
    remains intentionally outside the model.
    """
    result = df.copy()
    result["_cluster_identity"] = result.apply(_book_identity, axis=1)
    result = result.drop_duplicates("_cluster_identity", keep="first")
    return result.drop(columns=["_cluster_identity"])


def build_cluster_texts(df: pd.DataFrame) -> list[str]:
    """Build descriptive text used for semantic cluster discovery."""
    texts: list[str] = []

    for _, row in df.iterrows():
        subjects = parse_list_value(row.get("subjects"))
        subject_text = " ".join(
            clean
            for value in subjects
            if (clean := normalize_attribute(value))
        )

        title = str(row.get("title") or "")
        author = str(row.get("author") or "")
        description = str(row.get("description") or "")

        texts.append(
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Subjects: {subject_text}\n"
            f"Description: {description}"
        )

    return texts


def fit_kmeans_clusters(
    embeddings: np.ndarray,
    k: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans and return the model plus unit-normalized centroids."""
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(embeddings)

    centroids = model.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(norms, 1e-12, None)

    return model, centroids


def evaluate_cluster_solution(
    embeddings: np.ndarray,
    k: int,
    seeds: tuple[int, ...] = (42, 43, 44),
    n_init: int = DEFAULT_N_INIT,
) -> dict[str, float]:
    """
    Evaluate a clustering candidate on quality and stability.

    Silhouette measures within/between-cluster separation.
    ARI measures whether independently seeded fits recover similar
    assignments. Neither is treated as a semantic truth metric.
    """
    labels_by_seed: list[np.ndarray] = []
    silhouettes: list[float] = []
    min_sizes: list[int] = []

    for seed in seeds:
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=n_init,
        )
        labels = model.fit_predict(embeddings)
        labels_by_seed.append(labels)
        silhouettes.append(float(silhouette_score(embeddings, labels)))
        min_sizes.append(int(np.bincount(labels).min()))

    ari_values = [
        adjusted_rand_score(labels_by_seed[0], labels_by_seed[i])
        for i in range(1, len(labels_by_seed))
    ]

    return {
        "k": k,
        "silhouette_mean": float(np.mean(silhouettes)),
        "silhouette_std": float(np.std(silhouettes)),
        "stability_ari_mean": float(np.mean(ari_values)) if ari_values else 1.0,
        "min_cluster_size": int(min(min_sizes)),
        "min_cluster_pct": float(min(min_sizes) / len(embeddings)),
    }


def tune_cluster_count(
    embeddings: np.ndarray,
    config: ClusterConfig = ClusterConfig(),
) -> pd.DataFrame:
    """Evaluate a range of K values for the pooled semantic book space."""
    rows = []

    upper = min(config.max_k, len(embeddings) - 1)
    if upper < config.min_k:
        raise ValueError("Not enough books to evaluate the requested K range.")

    for k in range(config.min_k, upper + 1):
        rows.append(
            evaluate_cluster_solution(
                embeddings,
                k=k,
                seeds=(
                    config.random_state,
                    config.random_state + 1,
                    config.random_state + 2,
                ),
                n_init=config.n_init,
            )
        )

    return pd.DataFrame(rows)


def select_final_k(
    tuning: pd.DataFrame,
    min_stability: float = 0.70,
    min_cluster_pct: float = 0.01,
) -> int:
    """
    Select K using explicit guardrails rather than a hidden weighted score.

    1. Require reasonable assignment stability and non-tiny clusters.
    2. Among eligible solutions, choose the highest silhouette.
    3. Break ties toward the simpler model.
    4. If no solution satisfies both guardrails, choose the highest
       silhouette among the evaluated candidates and report the fallback.
    """
    eligible = tuning[
        (tuning["stability_ari_mean"] >= min_stability)
        & (tuning["min_cluster_pct"] >= min_cluster_pct)
    ].copy()

    candidates = eligible if not eligible.empty else tuning.copy()

    selected = candidates.sort_values(
        ["silhouette_mean", "k"],
        ascending=[False, True],
    ).iloc[0]

    return int(selected["k"])


def assign_clusters(
    embeddings: np.ndarray,
    k: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> tuple[np.ndarray, np.ndarray, KMeans]:
    """Fit final clustering and return labels, normalized centroids, model."""
    model, centroids = fit_kmeans_clusters(
        embeddings,
        k=k,
        random_state=random_state,
        n_init=n_init,
    )
    return model.labels_, centroids, model


def build_cluster_descriptions(
    df: pd.DataFrame,
    labels: np.ndarray,
    top_terms: int = 8,
) -> pd.DataFrame:
    """Create descriptive cluster labels from attributes, not generated genres."""
    work = df.copy()
    work["cluster_id"] = labels

    attribute_rows = []
    for _, row in work.iterrows():
        for attribute in row.get("attributes", []):
            attribute_rows.append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "attribute": attribute,
                }
            )

    if not attribute_rows:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "book_count",
                "top_attributes",
            ]
        )

    attrs = pd.DataFrame(attribute_rows)
    counts = (
        attrs.groupby(["cluster_id", "attribute"])
        .size()
        .reset_index(name="count")
    )

    rows = []
    for cluster_id, group in counts.groupby("cluster_id"):
        top = (
            group.sort_values(["count", "attribute"], ascending=[False, True])
            .head(top_terms)
        )

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "book_count": int((work["cluster_id"] == cluster_id).sum()),
                "top_attributes": top["attribute"].tolist(),
            }
        )

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def derive_reader_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add independent preference, exposure, and intent evidence streams.

    No arbitrary weights are applied.
    """
    result = df.copy()

    if "rating" in result.columns:
        ratings = pd.to_numeric(result["rating"], errors="coerce")
    else:
        ratings = pd.to_numeric(
            result.get("user_rating", pd.Series(index=result.index)),
            errors="coerce",
        )

    ratings = ratings.replace(0, np.nan)
    result["rating_numeric"] = ratings

    status = (
        result.get("reading_status", pd.Series(index=result.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Some experiment inputs may not yet have canonical reading_status.
    if status.eq("").all() and "shelves" in result.columns:
        shelves = result["shelves"].fillna("").astype(str).str.lower()
        status = np.where(
            shelves.str.contains("did-not-finish", regex=False),
            "did_not_finish",
            np.where(
                shelves.str.contains("currently-reading", regex=False),
                "currently_reading",
                np.where(
                    shelves.str.contains("to-read", regex=False),
                    "to_read",
                    np.where(
                        ratings.notna() & (ratings >= POSITIVE_RATING_MIN),
                        "read",
                        np.where(
                            ratings.notna() & (ratings <= NEGATIVE_RATING_MAX),
                            "read",
                            np.where(
                                pd.to_numeric(
                                    result.get("date_read", pd.Series(index=result.index)),
                                    errors="coerce",
                                ).notna(),
                                "read",
                                "unknown",
                            ),
                        ),
                    ),
                ),
            ),
        )
        status = pd.Series(status, index=result.index)

    result["reading_status_model_c"] = status

    result["preference_signal"] = np.select(
        [
            status.eq("did_not_finish"),
            ratings >= POSITIVE_RATING_MIN,
            ratings <= NEGATIVE_RATING_MAX,
        ],
        [
            "negative",
            "positive",
            "negative",
        ],
        default="neutral",
    )

    result["exposure_signal"] = np.where(
        status.isin(OBSERVED_STATUSES),
        "observed",
        "not_observed",
    )

    result["intent_signal"] = np.where(
        status.eq("to_read"),
        "intent",
        "no_intent",
    )

    return result


def aggregate_reader_attributes(
    df: pd.DataFrame,
    reader: str,
) -> pd.DataFrame:
    """
    Aggregate multi-label attributes into a reader-level Reading DNA table.

    Each attribute gets separate preference, exposure, and intent evidence.
    """
    evidence = derive_reader_evidence(df)

    rows: list[dict] = []

    for _, row in evidence.iterrows():
        attributes = row.get("attributes", [])
        if not isinstance(attributes, list):
            attributes = parse_list_value(attributes)

        for attribute in attributes:
            rows.append(
                {
                    "reader": reader,
                    "attribute": attribute,
                    "preference_signal": row["preference_signal"],
                    "exposure_signal": row["exposure_signal"],
                    "intent_signal": row["intent_signal"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "reader",
                "attribute",
                "book_count",
                "positive_count",
                "negative_count",
                "neutral_count",
                "observed_count",
                "intent_count",
                "preference_rate",
                "exposure_rate",
                "intent_rate",
            ]
        )

    long = pd.DataFrame(rows)

    grouped = (
        long.groupby(["reader", "attribute"])
        .agg(
            book_count=("attribute", "size"),
            positive_count=(
                "preference_signal",
                lambda x: int((x == "positive").sum()),
            ),
            negative_count=(
                "preference_signal",
                lambda x: int((x == "negative").sum()),
            ),
            neutral_count=(
                "preference_signal",
                lambda x: int((x == "neutral").sum()),
            ),
            observed_count=(
                "exposure_signal",
                lambda x: int((x == "observed").sum()),
            ),
            intent_count=(
                "intent_signal",
                lambda x: int((x == "intent").sum()),
            ),
        )
        .reset_index()
    )

    explicit = grouped["positive_count"] + grouped["negative_count"]
    grouped["preference_rate"] = np.where(
        explicit > 0,
        grouped["positive_count"] / explicit,
        np.nan,
    )

    grouped["exposure_rate"] = (
        grouped["observed_count"] / grouped["book_count"]
    )

    grouped["intent_rate"] = (
        grouped["intent_count"] / grouped["book_count"]
    )

    return grouped.sort_values(
        ["reader", "book_count", "attribute"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_attribute_combinations(
    df: pd.DataFrame,
    reader: str,
) -> pd.DataFrame:
    """Build reader-level attribute co-occurrence without collapsing labels."""
    rows: list[dict] = []

    evidence = derive_reader_evidence(df)

    for _, row in evidence.iterrows():
        attributes = sorted(
            {
                attr
                for attr in row.get("attributes", [])
                if attr
            }
        )

        for first, second in combinations(attributes, 2):
            rows.append(
                {
                    "reader": reader,
                    "attribute_1": first,
                    "attribute_2": second,
                    "preference_signal": row["preference_signal"],
                    "exposure_signal": row["exposure_signal"],
                    "intent_signal": row["intent_signal"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "reader",
                "attribute_1",
                "attribute_2",
                "book_count",
                "positive_count",
                "negative_count",
                "observed_count",
                "intent_count",
            ]
        )

    long = pd.DataFrame(rows)

    grouped = (
        long.groupby(["reader", "attribute_1", "attribute_2"])
        .agg(
            book_count=("reader", "size"),
            positive_count=(
                "preference_signal",
                lambda x: int((x == "positive").sum()),
            ),
            negative_count=(
                "preference_signal",
                lambda x: int((x == "negative").sum()),
            ),
            observed_count=(
                "exposure_signal",
                lambda x: int((x == "observed").sum()),
            ),
            intent_count=(
                "intent_signal",
                lambda x: int((x == "intent").sum()),
            ),
        )
        .reset_index()
    )

    return grouped.sort_values(
        ["reader", "book_count", "attribute_1", "attribute_2"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def build_reader_cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    reader: str,
) -> pd.DataFrame:
    """Summarize each semantic cluster through the reader's evidence streams."""
    evidence = derive_reader_evidence(df)
    evidence["cluster_id"] = labels

    grouped = (
        evidence.groupby("cluster_id")
        .agg(
            book_count=("cluster_id", "size"),
            positive_count=(
                "preference_signal",
                lambda x: int((x == "positive").sum()),
            ),
            negative_count=(
                "preference_signal",
                lambda x: int((x == "negative").sum()),
            ),
            observed_count=(
                "exposure_signal",
                lambda x: int((x == "observed").sum()),
            ),
            intent_count=(
                "intent_signal",
                lambda x: int((x == "intent").sum()),
            ),
        )
        .reset_index()
    )

    grouped.insert(0, "reader", reader)

    explicit = grouped["positive_count"] + grouped["negative_count"]
    grouped["preference_rate"] = np.where(
        explicit > 0,
        grouped["positive_count"] / explicit,
        np.nan,
    )

    grouped["exposure_rate"] = (
        grouped["observed_count"] / grouped["book_count"]
    )
    grouped["intent_rate"] = (
        grouped["intent_count"] / grouped["book_count"]
    )

    return grouped.sort_values("cluster_id").reset_index(drop=True)


def score_reader_attribute_strength(
    reader_attributes: pd.DataFrame,
    min_book_count: int = 3,
) -> pd.DataFrame:
    """
    Add descriptive strength flags for the UI.

    These are deliberately evidence labels, not a single opaque score.
    """
    result = reader_attributes.copy()

    result["evidence_level"] = np.select(
        [
            result["book_count"] < min_book_count,
            result["positive_count"] > 0,
            result["observed_count"] > 0,
        ],
        [
            "sparse",
            "preference_evidence",
            "exposure_only",
        ],
        default="intent_or_neutral",
    )

    return result
