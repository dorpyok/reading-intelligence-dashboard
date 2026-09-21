import pandas as pd

from src.analytics.preference_signals import derive_preference_signal


def test_positive_rating():
    row = pd.Series({
        "reading_status": "read",
        "rating": 5,
    })

    assert derive_preference_signal(row) == "positive"


def test_four_star_rating_is_positive():
    row = pd.Series({
        "reading_status": "read",
        "rating": 4,
    })

    assert derive_preference_signal(row) == "positive"


def test_three_star_rating_is_neutral():
    row = pd.Series({
        "reading_status": "read",
        "rating": 3,
    })

    assert derive_preference_signal(row) == "neutral"


def test_two_star_rating_is_negative():
    row = pd.Series({
        "reading_status": "read",
        "rating": 2,
    })

    assert derive_preference_signal(row) == "negative"


def test_one_star_rating_is_negative():
    row = pd.Series({
        "reading_status": "read",
        "rating": 1,
    })

    assert derive_preference_signal(row) == "negative"


def test_dnf_is_negative_without_rating():
    row = pd.Series({
        "reading_status": "did_not_finish",
        "rating": None,
    })

    assert derive_preference_signal(row) == "negative"


def test_unrated_read_is_neutral():
    row = pd.Series({
        "reading_status": "read",
        "rating": None,
    })

    assert derive_preference_signal(row) == "neutral"


def test_to_read_is_neutral():
    row = pd.Series({
        "reading_status": "to_read",
        "rating": None,
    })

    assert derive_preference_signal(row) == "neutral"


def test_currently_reading_is_neutral():
    row = pd.Series({
        "reading_status": "currently_reading",
        "rating": None,
    })

    assert derive_preference_signal(row) == "neutral"