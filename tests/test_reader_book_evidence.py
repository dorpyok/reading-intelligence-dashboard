import pandas as pd
import pytest

from src.analytics.reader_book_evidence import (
    aggregate_reader_book_evidence,
)


def make_evidence(**overrides):
    row = {
        "reader_id": "reader_1",
        "canonical_book_id": "book_1",
        "canonical_work_id": "work_1",
        "reading_status": "read",
        "user_rating": 5,
        "date_read": "2026-01-01",
        "is_read": True,
        "is_positive": True,
        "is_negative": False,
        "is_exposure": False,
        "is_tbr": False,
        "is_currently_reading": False,
        "is_dnf": False,
        "evidence_strength": "strong",
    }

    row.update(overrides)

    return pd.DataFrame([row])


def test_one_record_stays_one_relationship():
    evidence = make_evidence()

    result = aggregate_reader_book_evidence(evidence)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["reader_id"] == "reader_1"
    assert row["canonical_book_id"] == "book_1"
    assert row["canonical_work_id"] == "work_1"

    assert bool(row["is_read"]) is True
    assert bool(row["is_positive"]) is True
    assert bool(row["is_negative"]) is False
    assert bool(row["is_exposure"]) is False

    assert row["source_record_count"] == 1


def test_tbr_and_read_are_aggregated_into_one_relationship():
    evidence = pd.concat(
        [
            make_evidence(
                reading_status="to_read",
                user_rating=0,
                date_read=None,
                is_read=False,
                is_positive=False,
                is_negative=False,
                is_exposure=True,
                is_tbr=True,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="exposure",
            ),
            make_evidence(
                reading_status="read",
                user_rating=4,
                date_read="2026-01-01",
                is_read=True,
                is_positive=True,
                is_negative=False,
                is_exposure=False,
                is_tbr=False,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="strong",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["reading_status"] == "read"
    assert bool(row["is_read"]) is True
    assert bool(row["is_positive"]) is True
    assert bool(row["is_tbr"]) is True
    assert bool(row["is_exposure"]) is True

    assert row["source_record_count"] == 2
    assert row["max_user_rating"] == 4


def test_read_status_beats_tbr_status():
    evidence = pd.concat(
        [
            make_evidence(
                reading_status="to_read",
                user_rating=0,
                date_read=None,
                is_read=False,
                is_positive=False,
                is_negative=False,
                is_exposure=True,
                is_tbr=True,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="exposure",
            ),
            make_evidence(
                reading_status="read",
                user_rating=4,
                date_read="2026-01-01",
                is_read=True,
                is_positive=True,
                is_negative=False,
                is_exposure=False,
                is_tbr=False,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="strong",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert row["reading_status"] == "read"


def test_dnf_status_beats_read_status():
    evidence = pd.concat(
        [
            make_evidence(
                reading_status="read",
                user_rating=4,
                date_read="2026-01-01",
                is_read=True,
                is_positive=True,
                is_negative=False,
                is_exposure=False,
                is_tbr=False,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="strong",
            ),
            make_evidence(
                reading_status="did_not_finish",
                user_rating=0,
                date_read=None,
                is_read=True,
                is_positive=False,
                is_negative=True,
                is_exposure=False,
                is_tbr=False,
                is_currently_reading=False,
                is_dnf=True,
                evidence_strength="negative",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert row["reading_status"] == "did_not_finish"

def test_latest_date_is_selected():
    evidence = pd.concat(
        [
            make_evidence(
                date_read="2025-01-01",
            ),
            make_evidence(
                date_read="2026-02-15",
            ),
            make_evidence(
                date_read="2025-12-01",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert pd.Timestamp(row["latest_date_read"]).date() == pd.Timestamp(
        "2026-02-15"
    ).date()


def test_max_rating_is_selected():
    evidence = pd.concat(
        [
            make_evidence(user_rating=3),
            make_evidence(user_rating=5),
            make_evidence(user_rating=4),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert row["max_user_rating"] == 5


def test_multiple_exposure_records_remain_exposure_only():
    evidence = pd.concat(
        [
            make_evidence(
                reading_status="to_read",
                user_rating=0,
                date_read=None,
                is_read=False,
                is_positive=False,
                is_negative=False,
                is_exposure=True,
                is_tbr=True,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="exposure",
            ),
            make_evidence(
                reading_status="to_read",
                user_rating=0,
                date_read=None,
                is_read=False,
                is_positive=False,
                is_negative=False,
                is_exposure=True,
                is_tbr=True,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="exposure",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert bool(row["is_read"]) is False
    assert bool(row["is_positive"]) is False
    assert bool(row["is_negative"]) is False
    assert bool(row["is_exposure"]) is True
    assert bool(row["is_tbr"]) is True

    assert row["preference_signal"] == "exposure_only"
    assert row["source_record_count"] == 2


def test_read_without_rating_is_not_positive():
    evidence = make_evidence(
        reading_status="read",
        user_rating=0,
        date_read="2026-01-01",
        is_read=True,
        is_positive=False,
        is_negative=False,
        is_exposure=False,
        is_tbr=False,
        is_currently_reading=False,
        is_dnf=False,
        evidence_strength="moderate",
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert bool(row["is_read"]) is True
    assert bool(row["is_positive"]) is False
    assert bool(row["is_negative"]) is False

    assert row["preference_signal"] == "read_without_preference"


def test_positive_and_negative_evidence_is_conflicted():
    evidence = pd.concat(
        [
            make_evidence(
                reading_status="read",
                user_rating=5,
                date_read="2026-01-01",
                is_read=True,
                is_positive=True,
                is_negative=False,
                is_exposure=False,
                is_tbr=False,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="strong",
            ),
            make_evidence(
                reading_status="read",
                user_rating=2,
                date_read="2026-02-01",
                is_read=True,
                is_positive=False,
                is_negative=True,
                is_exposure=False,
                is_tbr=False,
                is_currently_reading=False,
                is_dnf=False,
                evidence_strength="negative",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert bool(row["is_positive"]) is True
    assert bool(row["is_negative"]) is True

    assert row["preference_signal"] == "conflicted"


def test_dnf_is_preserved():
    evidence = make_evidence(
        reading_status="did_not_finish",
        user_rating=0,
        date_read=None,
        is_read=True,
        is_positive=False,
        is_negative=True,
        is_exposure=False,
        is_tbr=False,
        is_currently_reading=False,
        is_dnf=True,
        evidence_strength="negative",
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert row["reading_status"] == "did_not_finish"

    assert bool(row["is_read"]) is True
    assert bool(row["is_negative"]) is True
    assert bool(row["is_dnf"]) is True

    assert row["preference_signal"] == "negative"


def test_record_counts_are_aggregated():
    evidence = pd.concat(
        [
            make_evidence(),
            make_evidence(),
            make_evidence(),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    row = result.iloc[0]

    assert row["source_record_count"] == 3


def test_multiple_reader_book_relationships_are_kept_separate():
    evidence = pd.concat(
        [
            make_evidence(
                reader_id="reader_1",
                canonical_book_id="book_1",
                canonical_work_id="work_1",
            ),
            make_evidence(
                reader_id="reader_1",
                canonical_book_id="book_2",
                canonical_work_id="work_2",
            ),
            make_evidence(
                reader_id="reader_2",
                canonical_book_id="book_1",
                canonical_work_id="work_1",
            ),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    assert len(result) == 3

    relationships = set(
        zip(
            result["reader_id"],
            result["canonical_book_id"],
        )
    )

    assert relationships == {
        ("reader_1", "book_1"),
        ("reader_1", "book_2"),
        ("reader_2", "book_1"),
    }


def test_multiple_work_ids_for_one_reader_book_fail():
    evidence = pd.concat(
        [
            make_evidence(
                canonical_work_id="work_1",
            ),
            make_evidence(
                canonical_work_id="work_2",
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="multiple canonical works"):
        aggregate_reader_book_evidence(evidence)


def test_missing_required_columns_fail():
    evidence = make_evidence()

    evidence = evidence.drop(columns=["canonical_work_id"])

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_reader_book_evidence(evidence)


def test_missing_reader_or_book_identifier_fails():
    evidence = make_evidence()

    evidence.loc[0, "reader_id"] = None

    with pytest.raises(ValueError, match="missing reader_id"):
        aggregate_reader_book_evidence(evidence)


def test_output_contains_no_duplicate_reader_book_pairs():
    evidence = pd.concat(
        [
            make_evidence(),
            make_evidence(),
            make_evidence(),
        ],
        ignore_index=True,
    )

    result = aggregate_reader_book_evidence(evidence)

    duplicates = result.duplicated(
        ["reader_id", "canonical_book_id"]
    )

    assert not duplicates.any()