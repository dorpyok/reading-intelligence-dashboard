import numpy as np

from src.analytics.book_similarity import (
    calculate_similarity_matrix,
    get_similar_books,
)


def test_similarity_matrix_is_square():
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    matrix = calculate_similarity_matrix(embeddings)

    assert matrix.shape == (3, 3)


def test_similarity_matrix_diagonal_is_one():
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    matrix = calculate_similarity_matrix(embeddings)

    assert np.allclose(
        np.diag(matrix),
        1.0,
    )


def test_most_similar_book_is_returned_first():
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    results = get_similar_books(
        book_index=0,
        embeddings=embeddings,
        top_n=2,
    )

    assert results[0][0] == 1


def test_queried_book_is_excluded():
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    results = get_similar_books(
        book_index=0,
        embeddings=embeddings,
        top_n=2,
    )

    indices = [index for index, score in results]

    assert 0 not in indices


def test_similarity_scores_are_descending():
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])

    results = get_similar_books(
        book_index=0,
        embeddings=embeddings,
        top_n=2,
    )

    scores = [score for index, score in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )