import numpy as np
import pandas as pd
import pytest

from src.analytics.reader_preference import (
    build_reader_preference_vector,
)


def test_reader_vector_is_mean_of_positive_embeddings():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    preference_signals = pd.Series(
        [
            "positive",
            "positive",
            "negative",
        ]
    )

    result = build_reader_preference_vector(
        embeddings,
        preference_signals,
    )

    expected = np.array(
        [1.0, 1.0]
    ) / np.sqrt(2)

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-7,
    )


def test_negative_and_neutral_books_are_excluded():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    preference_signals = pd.Series(
        [
            "positive",
            "negative",
            "neutral",
        ]
    )

    result = build_reader_preference_vector(
        embeddings,
        preference_signals,
    )

    expected = np.array([1.0, 0.0])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-7,
    )


def test_mismatched_lengths_raise_error():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    preference_signals = pd.Series(
        ["positive"]
    )

    with pytest.raises(ValueError):
        build_reader_preference_vector(
            embeddings,
            preference_signals,
        )


def test_no_positive_evidence_raises_error():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    preference_signals = pd.Series(
        [
            "negative",
            "neutral",
        ]
    )

    with pytest.raises(ValueError):
        build_reader_preference_vector(
            embeddings,
            preference_signals,
        )