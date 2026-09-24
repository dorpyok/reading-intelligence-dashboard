from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a 1D vector to unit length."""
    return normalize(vector.reshape(1, -1))[0]


def build_reader_preference_vector(
    embeddings: np.ndarray,
    preference_signals: pd.Series,
) -> np.ndarray:
    """
    Build a reader preference vector from positively preferred books.

    Model A:
        Only books with a positive preference signal contribute
        to the reader representation.

    Positive preference:
        - Rating >= 4

    Returns:
        Normalized reader preference vector.
    """
    if len(embeddings) != len(preference_signals):
        raise ValueError(
            "embeddings and preference_signals must contain the same number of rows."
        )

    positive_mask = (
        preference_signals.fillna("")
        .astype(str)
        .str.lower()
        .eq("positive")
        .to_numpy()
    )

    if not positive_mask.any():
        raise ValueError(
            "Cannot build a reader preference vector without positive preference evidence."
        )

    positive_embeddings = embeddings[positive_mask]
    reader_vector = positive_embeddings.mean(axis=0)

    return _normalize_vector(reader_vector)


def build_reader_exposure_vector(
    embeddings: np.ndarray,
    reading_behavior: pd.Series,
) -> np.ndarray:
    """
    Build a reader exposure vector from books the reader has actually engaged with.

    Exposure includes:
        - read
        - did_not_finish
        - currently_reading

    This vector represents the semantic space of books the reader
    has actually engaged with. It does not imply that the reader
    liked those books.

    Returns:
        Normalized reader exposure vector.
    """
    if len(embeddings) != len(reading_behavior):
        raise ValueError(
            "embeddings and reading_behavior must contain the same number of rows."
        )

    observed_mask = (
        reading_behavior.fillna("")
        .astype(str)
        .str.lower()
        .isin(
            {
                "read",
                "did_not_finish",
                "currently_reading",
            }
        )
        .to_numpy()
    )

    if not observed_mask.any():
        raise ValueError(
            "Cannot build a reader exposure vector without observed reading behavior."
        )

    observed_embeddings = embeddings[observed_mask]
    reader_vector = observed_embeddings.mean(axis=0)

    return _normalize_vector(reader_vector)


def build_reader_representation(
    embeddings: np.ndarray,
    preference_signals: pd.Series,
    reading_behavior: pd.Series,
) -> dict[str, np.ndarray]:
    """
    Build separate preference and exposure representations for a reader.

    Model B:
        - preference_vector represents explicit positive preference
        - exposure_vector represents observed reading behavior

    Keeping the vectors separate allows later experiments to test
    how much each evidence type should contribute to recommendations
    without hard-coding a weighting assumption now.

    Returns:
        Dictionary containing:
            - preference_vector
            - exposure_vector
    """
    if not (
        len(embeddings)
        == len(preference_signals)
        == len(reading_behavior)
    ):
        raise ValueError(
            "embeddings, preference_signals, and reading_behavior "
            "must contain the same number of rows."
        )

    return {
        "preference_vector": build_reader_preference_vector(
            embeddings,
            preference_signals,
        ),
        "exposure_vector": build_reader_exposure_vector(
            embeddings,
            reading_behavior,
        ),
    }