import pandas as pd
import pytest

from src.model.work_mapping import (
    build_canonical_work_mapping,
    validate_work_mapping,
)


AUDIT_COLUMNS = [
    "canonical_book_id_a",
    "canonical_book_id_b",
    "confidence",
]


def make_book(book_id):
    return {
        "canonical_book_id": book_id,
    }


def make_audit(rows=None):
    """
    Create a reconciliation audit with the correct schema.

    This intentionally preserves the expected columns even when the
    audit contains zero candidate pairs.
    """
    return pd.DataFrame(
        rows if rows is not None else [],
        columns=AUDIT_COLUMNS,
    )


def test_every_book_gets_a_work_id():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
            make_book("c"),
        ]
    )

    audit = make_audit()

    mapping = build_canonical_work_mapping(
        books,
        audit,
    )

    assert len(mapping) == 3
    assert mapping["canonical_book_id"].nunique() == 3
    assert mapping["canonical_work_id"].notna().all()


def test_standalone_books_get_distinct_work_ids():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    audit = make_audit()

    mapping = build_canonical_work_mapping(
        books,
        audit,
    )

    assert mapping["canonical_work_id"].nunique() == 2
    assert set(mapping["work_mapping_status"]) == {"standalone"}


def test_high_confidence_pair_shares_work_id():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
            make_book("c"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            }
        ]
    )

    mapping = build_canonical_work_mapping(
        books,
        audit,
    )

    row_a = mapping.loc[
        mapping["canonical_book_id"] == "a"
    ].iloc[0]

    row_b = mapping.loc[
        mapping["canonical_book_id"] == "b"
    ].iloc[0]

    row_c = mapping.loc[
        mapping["canonical_book_id"] == "c"
    ].iloc[0]

    assert (
        row_a["canonical_work_id"]
        == row_b["canonical_work_id"]
    )

    assert row_a["work_mapping_status"] == "reconciled"
    assert row_b["work_mapping_status"] == "reconciled"

    assert (
        row_c["canonical_work_id"]
        != row_a["canonical_work_id"]
    )

    assert row_c["work_mapping_status"] == "standalone"


def test_medium_confidence_does_not_merge_when_high_required():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "medium",
            }
        ]
    )

    mapping = build_canonical_work_mapping(
        books,
        audit,
        minimum_confidence="high",
    )

    assert mapping["canonical_work_id"].nunique() == 2
    assert set(mapping["work_mapping_status"]) == {"standalone"}


def test_low_confidence_does_not_merge_when_high_required():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "low",
            }
        ]
    )

    mapping = build_canonical_work_mapping(
        books,
        audit,
        minimum_confidence="high",
    )

    assert mapping["canonical_work_id"].nunique() == 2
    assert set(mapping["work_mapping_status"]) == {"standalone"}


def test_transitive_relationship_creates_one_work():
    """
    If A matches B and B matches C, all three books should resolve
    to the same canonical work.
    """
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
            make_book("c"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            },
            {
                "canonical_book_id_a": "b",
                "canonical_book_id_b": "c",
                "confidence": "high",
            },
        ]
    )

    mapping = build_canonical_work_mapping(
        books,
        audit,
    )

    assert mapping["canonical_work_id"].nunique() == 1
    assert set(mapping["work_mapping_status"]) == {"reconciled"}


def test_unconnected_book_remains_standalone():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
            make_book("c"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            }
        ]
    )

    mapping = build_canonical_work_mapping(
        books,
        audit,
    )

    row_c = mapping.loc[
        mapping["canonical_book_id"] == "c"
    ].iloc[0]

    assert row_c["work_mapping_status"] == "standalone"

    reconciled_work_id = mapping.loc[
        mapping["canonical_book_id"] == "a",
        "canonical_work_id",
    ].iloc[0]

    assert row_c["canonical_work_id"] != reconciled_work_id


def test_work_ids_are_deterministic():
    books = pd.DataFrame(
        [
            make_book("b"),
            make_book("a"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            }
        ]
    )

    mapping_one = build_canonical_work_mapping(
        books,
        audit,
    )

    mapping_two = build_canonical_work_mapping(
        books,
        audit,
    )

    first = (
        mapping_one
        .sort_values("canonical_book_id")
        .reset_index(drop=True)
    )

    second = (
        mapping_two
        .sort_values("canonical_book_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_work_id_is_stable_when_book_order_changes():
    """
    Changing dataframe row order should not change the work identity.
    """
    books_one = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    books_two = pd.DataFrame(
        [
            make_book("b"),
            make_book("a"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            }
        ]
    )

    mapping_one = build_canonical_work_mapping(
        books_one,
        audit,
    )

    mapping_two = build_canonical_work_mapping(
        books_two,
        audit,
    )

    work_one = mapping_one.loc[
        mapping_one["canonical_book_id"] == "a",
        "canonical_work_id",
    ].iloc[0]

    work_two = mapping_two.loc[
        mapping_two["canonical_book_id"] == "a",
        "canonical_work_id",
    ].iloc[0]

    assert work_one == work_two


def test_unknown_book_in_audit_raises_error():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "missing",
                "confidence": "high",
            }
        ]
    )

    with pytest.raises(ValueError, match="unknown"):
        build_canonical_work_mapping(
            books,
            audit,
        )


def test_invalid_minimum_confidence_raises_error():
    books = pd.DataFrame(
        [
            make_book("a"),
        ]
    )

    audit = make_audit()

    with pytest.raises(
        ValueError,
        match="minimum_confidence",
    ):
        build_canonical_work_mapping(
            books,
            audit,
            minimum_confidence="invalid",
        )


def test_validate_work_mapping():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    audit = make_audit(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            }
        ]
    )

    mapping = build_canonical_work_mapping(
        books,
        audit,
    )

    validate_work_mapping(
        mapping,
        books,
    )


def test_validate_work_mapping_detects_missing_book():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    mapping = pd.DataFrame(
        [
            {
                "canonical_book_id": "a",
                "canonical_work_id": "work_a",
                "work_mapping_status": "standalone",
                "work_group_root": "a",
            }
        ]
    )

    with pytest.raises(ValueError, match="exactly"):
        validate_work_mapping(
            mapping,
            books,
        )


def test_validate_work_mapping_detects_duplicate_book_mapping():
    books = pd.DataFrame(
        [
            make_book("a"),
            make_book("b"),
        ]
    )

    mapping = pd.DataFrame(
        [
            {
                "canonical_book_id": "a",
                "canonical_work_id": "work_a",
                "work_mapping_status": "standalone",
                "work_group_root": "a",
            },
            {
                "canonical_book_id": "a",
                "canonical_work_id": "work_b",
                "work_mapping_status": "standalone",
                "work_group_root": "a",
            },
            {
                "canonical_book_id": "b",
                "canonical_work_id": "work_c",
                "work_mapping_status": "standalone",
                "work_group_root": "b",
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_work_mapping(
            mapping,
            books,
        )


def test_validate_work_mapping_detects_invalid_status():
    books = pd.DataFrame(
        [
            make_book("a"),
        ]
    )

    mapping = pd.DataFrame(
        [
            {
                "canonical_book_id": "a",
                "canonical_work_id": "work_a",
                "work_mapping_status": "wrong_status",
                "work_group_root": "a",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Invalid work_mapping_status",
    ):
        validate_work_mapping(
            mapping,
            books,
        )