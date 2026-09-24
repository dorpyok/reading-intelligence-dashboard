import pandas as pd

from src.analytics.preference_evidence import (
    build_preference_evidence,
)


def test_build_preference_evidence_combines_signals():
    df = pd.DataFrame(
        {
            "rating": [5, 3, 2, None, None],
            "reading_status": [
                "read",
                "read",
                "read",
                "read",
                "to_read",
            ],
        }
    )

    result = build_preference_evidence(df)

    assert list(result["preference_signal"]) == [
        "positive",
        "neutral",
        "negative",
        "neutral",
        "neutral",
    ]

    assert list(result["reading_behavior"]) == [
        "read",
        "read",
        "read",
        "read",
        "to_read",
    ]

    assert list(result["reading_exposure"]) == [
        "observed",
        "observed",
        "observed",
        "observed",
        "not_observed",
    ]


def test_unrated_read_book_has_behavior_without_positive_preference():
    df = pd.DataFrame(
        {
            "rating": [None],
            "reading_status": ["read"],
        }
    )

    result = build_preference_evidence(df)

    assert result.loc[0, "preference_signal"] == "neutral"
    assert result.loc[0, "reading_behavior"] == "read"
    assert result.loc[0, "reading_exposure"] == "observed"


def test_dnf_is_negative_and_observed():
    df = pd.DataFrame(
        {
            "rating": [None],
            "reading_status": ["did_not_finish"],
        }
    )

    result = build_preference_evidence(df)

    assert result.loc[0, "preference_signal"] == "negative"
    assert result.loc[0, "reading_behavior"] == "did_not_finish"
    assert result.loc[0, "reading_exposure"] == "observed"