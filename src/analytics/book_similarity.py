from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate pairwise cosine similarity between book embeddings.

    Returns:
        A square matrix where each cell contains the similarity
        between two books.
    """

    return cosine_similarity(embeddings)


def get_similar_books(
    book_index: int,
    embeddings: np.ndarray,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """
    Return the most semantically similar books to a given book.

    The queried book itself is excluded from the results.

    Returns:
        List of (book_index, similarity_score) tuples,
        ordered from most similar to least similar.
    """

    similarities = cosine_similarity(
        embeddings[book_index].reshape(1, -1),
        embeddings,
    )[0]

    similarities[book_index] = -1

    nearest_indices = np.argsort(similarities)[::-1][:top_n]

    return [
        (
            int(index),
            float(similarities[index]),
        )
        for index in nearest_indices
    ]