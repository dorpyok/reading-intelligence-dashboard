import pandas as pd
import pytest

from src.model.work_reconciliation import (
    build_normalized_identity_columns,
    build_reconciliation_audit,
    build_work_groups_from_confirmed_pairs,
    generate_candidate_pairs,
    jaccard_similarity,
    normalize_base_title,
    normalize_identifier,
    normalize_isbn,
    normalize_text,
    normalize_title,
    reconcile_pair,
)


def make_book(
    book_id,
    title,
    author,
    isbn=None,
    description="",
    work_id=None,
    edition_id=None,
):
    return {
        "canonical_book_id": book_id,
        "title": title,
        "author": author,
        "isbn": isbn,
        "pages": None,
        "publication_year": None,
        "description": description,
        "cover_url": None,
        "openlibrary_work_id": work_id,
        "openlibrary_edition_id": edition_id,
        "identity_method": "title_author",
        "identity_confidence": "medium",
        "subjects": "[]",
        "subject_people": "[]",
        "subject_places": "[]",
        "subject_times": "[]",
    }


def test_normalize_text_and_title_variants():
    assert (
        normalize_text("The Ex Hex (The Ex Hex, #1)")
        == "the ex hex the ex hex 1"
    )
    assert (
        normalize_title("The Ex Hex (The Ex Hex, #1)")
        == "the ex hex the ex hex 1"
    )
    assert (
        normalize_base_title("The Ex Hex (The Ex Hex, #1)")
        == "the ex hex"
    )
    assert normalize_base_title("Nettle & Bone") == "nettle bone"


def test_normalize_isbn():
    assert normalize_isbn("978-0-316-11737-8") == "9780316117378"
    assert normalize_isbn("9780316117378") == "9780316117378"


def test_normalize_identifier():
    assert normalize_identifier(" ol123w ") == "OL123W"
    assert normalize_identifier(None) == ""


def test_jaccard_similarity():
    assert jaccard_similarity("one two three", "one two three") == 1.0
    assert jaccard_similarity("one two", "three four") == 0.0


def test_jaccard_similarity_handles_empty_text():
    assert jaccard_similarity("", "something") == 0.0
    assert jaccard_similarity("", "") == 0.0


def test_normalized_identity_columns():
    df = pd.DataFrame(
        [
            make_book(
                "a",
                "The Ex Hex (The Ex Hex, #1)",
                "Erin Sterling",
                isbn="123",
                work_id="OL123W",
                edition_id="OL123M",
            )
        ]
    )

    result = build_normalized_identity_columns(df)

    assert (
        result.loc[0, "normalized_title"]
        == "the ex hex the ex hex 1"
    )
    assert (
        result.loc[0, "normalized_base_title"]
        == "the ex hex"
    )
    assert (
        result.loc[0, "normalized_author"]
        == "erin sterling"
    )
    assert result.loc[0, "normalized_isbn"] == "123"
    assert result.loc[0, "normalized_work_id"] == "OL123W"
    assert result.loc[0, "normalized_edition_id"] == "OL123M"


def test_generate_candidate_pairs_blocks_same_title_author():
    df = pd.DataFrame(
        [
            make_book("a", "The Ex Hex", "Erin Sterling"),
            make_book("b", "The Ex Hex", "Erin Sterling"),
            make_book("c", "Nettle & Bone", "T. Kingfisher"),
        ]
    )

    pairs = generate_candidate_pairs(df)

    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs
    assert ("b", "c") not in pairs


def test_generate_candidate_pairs_does_not_self_match():
    df = pd.DataFrame(
        [
            make_book("a", "The Ex Hex", "Erin Sterling"),
        ]
    )

    pairs = generate_candidate_pairs(df)

    assert pairs == []


def test_reconcile_pair_high_confidence_same_work():
    row_a = pd.Series(
        make_book(
            "a",
            "The Ex Hex",
            "Erin Sterling",
            isbn="123",
            description="A witch casts a spell.",
            work_id="OL123W",
            edition_id="OL123M",
        )
    )

    row_b = pd.Series(
        make_book(
            "b",
            "The Ex Hex",
            "Erin Sterling",
            isbn="456",
            description="A witch casts a spell.",
            work_id="OL123W",
            edition_id="OL456M",
        )
    )

    result = reconcile_pair(row_a, row_b)

    assert result.title_match is True
    assert result.base_title_match is True
    assert result.author_match is True
    assert result.work_id_match is True
    assert result.edition_id_match is False
    assert result.description_similarity == 1.0
    assert result.confidence == "high"


def test_reconcile_pair_same_title_author_and_description():
    row_a = pd.Series(
        make_book(
            "a",
            "Yes, Chef",
            "Grace Reilly",
            description="A chef falls in love.",
        )
    )

    row_b = pd.Series(
        make_book(
            "b",
            "Yes, Chef",
            "Grace Reilly",
            description="A chef falls in love.",
        )
    )

    result = reconcile_pair(row_a, row_b)

    assert result.title_match is True
    assert result.base_title_match is True
    assert result.author_match is True
    assert result.description_similarity == 1.0
    assert result.confidence == "high"


def test_reconcile_pair_different_books_low_confidence():
    row_a = pd.Series(
        make_book(
            "a",
            "The Ex Hex",
            "Erin Sterling",
            description="A witch casts a spell.",
        )
    )

    row_b = pd.Series(
        make_book(
            "b",
            "Nettle & Bone",
            "T. Kingfisher",
            description="A princess completes impossible tasks.",
        )
    )

    result = reconcile_pair(row_a, row_b)

    assert result.title_match is False
    assert result.base_title_match is False
    assert result.author_match is False
    assert result.work_id_match is False
    assert result.confidence == "low"


def test_reconcile_pair_distinguishes_full_and_base_title():
    row_a = pd.Series(
        make_book(
            "a",
            "The Last Hope (Maggie Hope, #11)",
            "Susan Elia MacNeal",
            description="A mystery involving Maggie Hope.",
        )
    )

    row_b = pd.Series(
        make_book(
            "b",
            "The Last Hope (Maggie Hope Mystery, #11)",
            "Susan Elia MacNeal",
            description="A mystery involving Maggie Hope.",
        )
    )

    result = reconcile_pair(row_a, row_b)

    assert result.title_match is False
    assert result.base_title_match is True
    assert result.author_match is True
    assert result.description_similarity == 1.0
    assert result.confidence == "high"


def test_build_reconciliation_audit():
    df = pd.DataFrame(
        [
            make_book(
                "a",
                "The Ex Hex",
                "Erin Sterling",
                work_id="OL123W",
                description="A witch casts a spell.",
            ),
            make_book(
                "b",
                "The Ex Hex",
                "Erin Sterling",
                work_id="OL123W",
                description="A witch casts a spell.",
            ),
            make_book(
                "c",
                "Nettle & Bone",
                "T. Kingfisher",
            ),
        ]
    )

    audit = build_reconciliation_audit(df)

    assert len(audit) == 1
    assert audit.loc[0, "work_id_match"]
    assert audit.loc[0, "confidence"] == "high"


def test_build_reconciliation_audit_requires_unique_ids():
    df = pd.DataFrame(
        [
            make_book("a", "Book A", "Author"),
            make_book("a", "Book B", "Author"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_reconciliation_audit(df)


def test_build_work_groups_from_confirmed_pairs():
    audit = pd.DataFrame(
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
            {
                "canonical_book_id_a": "d",
                "canonical_book_id_b": "e",
                "confidence": "medium",
            },
        ]
    )

    groups = build_work_groups_from_confirmed_pairs(
        audit,
        minimum_confidence="high",
    )

    assert groups["a"] == groups["b"]
    assert groups["b"] == groups["c"]
    assert "d" not in groups
    assert "e" not in groups


def test_build_work_groups_rejects_invalid_confidence():
    audit = pd.DataFrame(
        [
            {
                "canonical_book_id_a": "a",
                "canonical_book_id_b": "b",
                "confidence": "high",
            }
        ]
    )

    with pytest.raises(ValueError):
        build_work_groups_from_confirmed_pairs(
            audit,
            minimum_confidence="invalid",
        )