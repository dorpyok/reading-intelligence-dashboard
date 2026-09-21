from __future__ import annotations

import hdbscan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import umap


def reduce_for_clustering(
    embeddings: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce semantic embeddings to a lower-dimensional representation
    suitable for clustering.
    """

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)


def cluster_books(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int = 3,
) -> hdbscan.HDBSCAN:
    """
    Discover naturally occurring book clusters using HDBSCAN.
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    clusterer.fit(reduced_embeddings)

    return clusterer


def reduce_for_visualization(
    embeddings: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create a two-dimensional representation for visualization.
    """

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)


def get_cluster_keywords(
    texts: list[str],
    cluster_labels: np.ndarray,
    top_n: int = 10,
) -> dict[int, list[str]]:
    """
    Identify terms that are particularly representative of each cluster.

    These are descriptive keywords, not authoritative genre labels.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    keywords = {}

    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue

        indices = np.where(cluster_labels == cluster_id)[0]

        cluster_scores = matrix[indices].mean(axis=0)
        cluster_scores = np.asarray(cluster_scores).ravel()

        top_indices = cluster_scores.argsort()[-top_n:][::-1]

        keywords[cluster_id] = feature_names[top_indices].tolist()

    return keywords