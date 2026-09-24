import pandas as pd

from src.analytics.reading_exposure import (
    add_reading_exposure,
    derive_reading_exposure,
)


def test_read_book_is_observed():
    row = pd.Series(
        {
            "reading_status": "read",
        }
    )

    assert derive_reading_exposure(row) == "observed"


def test_dnf_book_is_observed():
    row = pd.Series(
        {
            "reading_status": "did_not_finish",
        }
    )

    assert derive_reading_exposure(row) == "observed"


def test_currently_reading_is_observed():
    row = pd.Series(
        {
            "reading_status": "currently_reading",
        }
    )

    assert derive_reading_exposure(row) == "observed"


def test_to_read_is_not_observed():
    row = pd.Series(
        {
            "reading_status": "to_read",
        }
    )

    assert (
        derive_reading_exposure(row)
        == "not_observed"
    )


def test_unknown_is_not_observed():
    row = pd.Series(
        {
            "reading_status": "unknown",
        }
    )

    assert (
        derive_reading_exposure(row)
        == "not_observed"
    )


def test_add_reading_exposure():
    df = pd.DataFrame(
        {
            "reading_status": [
                "read",
                "did_not_finish",
                "currently_reading",
                "to_read",
                "unknown",
            ]
        }
    )

    result = add_reading_exposure(df)

    assert list(result["reading_exposure"]) == [
        "observed",
        "observed",
        "observed",
        "not_observed",
        "not_observed",
    ]