import pandas as pd

from src.analytics.reading_behavior import (
    add_reading_behavior,
    derive_reading_behavior,
)


def test_read_status_is_read():
    row = pd.Series(
        {
            "reading_status": "read",
        }
    )

    assert derive_reading_behavior(row) == "read"


def test_unrated_read_book_is_still_read_behavior():
    row = pd.Series(
        {
            "reading_status": "read",
            "rating": None,
        }
    )

    assert derive_reading_behavior(row) == "read"


def test_dnf_is_reading_behavior():
    row = pd.Series(
        {
            "reading_status": "did_not_finish",
        }
    )

    assert (
        derive_reading_behavior(row)
        == "did_not_finish"
    )


def test_to_read_is_not_read_behavior():
    row = pd.Series(
        {
            "reading_status": "to_read",
        }
    )

    assert (
        derive_reading_behavior(row)
        == "to_read"
    )


def test_currently_reading_is_distinct():
    row = pd.Series(
        {
            "reading_status": "currently_reading",
        }
    )

    assert (
        derive_reading_behavior(row)
        == "currently_reading"
    )


def test_unknown_status_is_unknown():
    row = pd.Series(
        {
            "reading_status": "unknown",
        }
    )

    assert (
        derive_reading_behavior(row)
        == "unknown"
    )


def test_add_reading_behavior():
    df = pd.DataFrame(
        {
            "reading_status": [
                "read",
                "to_read",
                "did_not_finish",
            ]
        }
    )

    result = add_reading_behavior(df)

    assert list(result["reading_behavior"]) == [
        "read",
        "to_read",
        "did_not_finish",
    ]