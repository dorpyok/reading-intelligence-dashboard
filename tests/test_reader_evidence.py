import pandas as pd
import pytest

from src.analytics.reader_evidence import (
    build_reader_evidence,
    validate_reader_books,
)


REQUIRED_COLUMNS = [
    "reader_id",
    "canonical_work_id",
    "canonical_book_id",
    "reading_status",
    "user_rating",
    "date_read",
]


def make_book(
    reader_id="reader_1",
    work_id="work_1",
    book_id="book_1",
    status="unknown",
    rating=None,
    date_read=None,
):
    return {
        "reader_id": reader_id,
        "canonical_work_id": work_id,
        "canonical_book_id": book_id,
        "reading_status": status,
        "user_rating": rating,
        "date_read": date_read,
    }


def test_required_columns_are_present():
    data = pd.DataFrame([make_book()])
    validate_reader_books(data)


def test_missing_required_column_raises():
    data = pd.DataFrame(
        [
            {
                "reader_id": "reader_1",
                "canonical_work_id": "work_1",
                "canonical_book_id": "book_1",
                "reading_status": "read",
                "user_rating": 5,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_reader_books(data)


def test_invalid_status_raises():
    data = pd.DataFrame(
        [
            make_book(
                status="finished_somehow",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="invalid reading_status",
    ):
        validate_reader_books(data)


def test_five_star_rating_is_strong_positive():
    data = pd.DataFrame(
        [
            make_book(
                status="read",
                rating=5,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == True
    assert row["is_positive"] == True
    assert row["is_negative"] == False
    assert row["is_exposure"] == False
    assert row["evidence_strength"] == "strong"


def test_three_star_rating_is_positive():
    data = pd.DataFrame(
        [
            make_book(
                status="read",
                rating=3,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == True
    assert row["is_positive"] == True
    assert row["is_negative"] == False
    assert row["evidence_strength"] == "strong"


def test_two_star_rating_is_negative():
    data = pd.DataFrame(
        [
            make_book(
                status="read",
                rating=2,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == True
    assert row["is_positive"] == False
    assert row["is_negative"] == True
    assert row["is_exposure"] == False
    assert row["evidence_strength"] == "negative"


def test_one_star_rating_is_negative():
    data = pd.DataFrame(
        [
            make_book(
                status="read",
                rating=1,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == True
    assert row["is_positive"] == False
    assert row["is_negative"] == True
    assert row["evidence_strength"] == "negative"


def test_read_status_without_rating_is_moderate_but_neutral():
    data = pd.DataFrame(
        [
            make_book(
                status="read",
                rating=None,
                date_read=None,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == True
    assert row["is_positive"] == False
    assert row["is_negative"] == False
    assert row["is_exposure"] == False
    assert row["evidence_strength"] == "moderate"


def test_read_date_creates_read_behavior_not_preference():
    data = pd.DataFrame(
        [
            make_book(
                status="unknown",
                rating=None,
                date_read="2025-08-15",
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == True
    assert row["is_positive"] == False
    assert row["is_negative"] == False
    assert row["is_exposure"] == False
    assert row["evidence_strength"] == "moderate"


def test_dnf_is_negative():
    data = pd.DataFrame(
        [
            make_book(
                status="did_not_finish",
                rating=None,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == False
    assert row["is_positive"] == False
    assert row["is_negative"] == True
    assert row["is_dnf"] == True
    assert row["evidence_strength"] == "negative"


def test_dnf_overrides_positive_rating_for_strength():
    data = pd.DataFrame(
        [
            make_book(
                status="did_not_finish",
                rating=5,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_negative"] == True
    assert row["is_positive"] == False
    assert row["evidence_strength"] == "negative"


def test_tbr_is_exposure_not_preference():
    data = pd.DataFrame(
        [
            make_book(
                status="to_read",
                rating=None,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_tbr"] == True
    assert row["is_exposure"] == True
    assert row["is_read"] == False
    assert row["is_positive"] == False
    assert row["is_negative"] == False
    assert row["evidence_strength"] == "exposure"


def test_currently_reading_is_exposure():
    data = pd.DataFrame(
        [
            make_book(
                status="currently_reading",
                rating=None,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_currently_reading"] == True
    assert row["is_exposure"] == True
    assert row["is_read"] == False
    assert row["is_positive"] == False
    assert row["is_negative"] == False
    assert row["evidence_strength"] == "exposure"


def test_unknown_without_evidence_remains_unknown():
    data = pd.DataFrame(
        [
            make_book(
                status="unknown",
                rating=None,
                date_read=None,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == False
    assert row["is_positive"] == False
    assert row["is_negative"] == False
    assert row["is_exposure"] == False
    assert row["evidence_strength"] == "unknown"


def test_zero_rating_is_not_positive_or_negative():
    data = pd.DataFrame(
        [
            make_book(
                status="unknown",
                rating=0,
            )
        ]
    )

    evidence = build_reader_evidence(data)
    row = evidence.iloc[0]

    assert row["is_read"] == False
    assert row["is_positive"] == False
    assert row["is_negative"] == False
    assert row["evidence_strength"] == "unknown"


def test_all_input_records_are_preserved():
    data = pd.DataFrame(
        [
            make_book(
                book_id="book_1",
                status="read",
                rating=5,
            ),
            make_book(
                book_id="book_2",
                status="to_read",
            ),
            make_book(
                book_id="book_3",
                status="unknown",
            ),
            make_book(
                book_id="book_4",
                status="did_not_finish",
            ),
        ]
    )

    evidence = build_reader_evidence(data)

    assert len(evidence) == len(data)

    assert set(
        evidence["canonical_book_id"]
    ) == {
        "book_1",
        "book_2",
        "book_3",
        "book_4",
    }


def test_multiple_readers_are_preserved():
    data = pd.DataFrame(
        [
            make_book(
                reader_id="reader_1",
                book_id="book_1",
                status="read",
                rating=5,
            ),
            make_book(
                reader_id="reader_2",
                book_id="book_2",
                status="read",
                rating=4,
            ),
        ]
    )

    evidence = build_reader_evidence(data)

    assert set(
        evidence["reader_id"]
    ) == {
        "reader_1",
        "reader_2",
    }


def test_empty_dataframe_returns_valid_schema():
    data = pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )

    evidence = build_reader_evidence(data)

    assert evidence.empty

    expected_columns = [
        "reader_id",
        "canonical_work_id",
        "canonical_book_id",
        "reading_status",
        "user_rating",
        "date_read",
        "is_read",
        "is_positive",
        "is_negative",
        "is_exposure",
        "is_tbr",
        "is_currently_reading",
        "is_dnf",
        "evidence_strength",
    ]

    assert list(evidence.columns) == expected_columns


def test_duplicate_records_are_not_silently_removed():
    data = pd.DataFrame(
        [
            make_book(
                book_id="book_1",
                status="read",
                rating=5,
            ),
            make_book(
                book_id="book_1",
                status="read",
                rating=5,
            ),
        ]
    )

    evidence = build_reader_evidence(data)

    assert len(evidence) == 2