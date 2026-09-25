import numpy as np

from scripts.run_recommendation_experiment import (
    calculate_recall_at_k,
    calculate_rank_percentiles,
    rank_candidates,
)


def test_recall_at_k():
    ranked_indices = np.array(
        [4, 2, 7, 1, 9, 3, 5, 0]
    )

    held_out_indices = {4, 7}

    recall = calculate_recall_at_k(
        ranked_indices,
        held_out_indices,
        k=3,
    )

    assert recall == 1.0


def test_recall_at_k_partial():
    ranked_indices = np.array(
        [4, 2, 7, 1, 9, 3, 5, 0]
    )

    held_out_indices = {4, 7}

    recall = calculate_recall_at_k(
        ranked_indices,
        held_out_indices,
        k=2,
    )

    assert recall == 0.5


def test_rank_candidates_excludes_training_books():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    preference_vector = np.array(
        [1.0, 0.0]
    )

    ranked = rank_candidates(
        preference_vector=preference_vector,
        embeddings=embeddings,
        excluded_indices={0},
    )

    assert 0 not in ranked


def test_rank_percentiles():
    ranked_indices = np.array(
        [0, 1, 2, 3, 4]
    )

    held_out_indices = np.array(
        [0, 4]
    )

    percentiles = calculate_rank_percentiles(
        ranked_indices,
        held_out_indices,
    )

    assert percentiles[0] == 1.0
    assert percentiles[1] == 0.2