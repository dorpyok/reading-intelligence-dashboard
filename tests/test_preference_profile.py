import pandas as pd

from src.analytics.preference_profile import (
    build_preference_profile,
)


def test_preference_profile_counts_ratings():
    df = pd.DataFrame(
        {
            "rating": [5, 4, 3, 2, 1, None],
            "reading_status": [
                "read",
                "read",
                "read",
                "read",
                "read",
                "read",
            ],
        }
    )

    profile = build_preference_profile(df)

    assert profile["total_books"] == 6
    assert profile["rated_books"] == 5
    assert profile["unrated_books"] == 1
    assert profile["positive_books"] == 2
    assert profile["negative_books"] == 2
    assert profile["neutral_books"] == 1


def test_dnf_counts_as_negative():
    df = pd.DataFrame(
        {
            "rating": [None],
            "reading_status": ["did_not_finish"],
        }
    )

    profile = build_preference_profile(df)

    assert profile["dnf_books"] == 1
    assert profile["negative_books"] == 1


def test_unrated_books_are_not_negative():
    df = pd.DataFrame(
        {
            "rating": [None, None],
            "reading_status": ["read", "to_read"],
        }
    )

    profile = build_preference_profile(df)

    assert profile["unrated_books"] == 2
    assert profile["negative_books"] == 0
    assert profile["positive_books"] == 0