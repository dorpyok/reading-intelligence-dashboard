from src.model.book_identity import (
    build_canonical_book_id,
    build_title_author_key,
    normalize_author,
    normalize_isbn,
    normalize_openlibrary_id,
    normalize_title,
)


def test_normalize_title():
    assert normalize_title(
        "The Great Gatsby!"
    ) == "the great gatsby"


def test_normalize_title_removes_accents():
    assert normalize_title(
        "José and the Amazing Book"
    ) == "jose and the amazing book"


def test_normalize_author():
    assert normalize_author(
        "Suzanne Collins"
    ) == "suzanne collins"


def test_normalize_isbn():
    assert normalize_isbn(
        "978-0-06-112008-4"
    ) == "9780061120084"


def test_normalize_isbn_handles_isbn10():
    assert normalize_isbn(
        "0-451-52953-7"
    ) == "0451529537"


def test_normalize_openlibrary_work_id():
    assert normalize_openlibrary_id(
        "OL123456W"
    ) == "OL123456W"


def test_normalize_openlibrary_work_url():
    assert normalize_openlibrary_id(
        "https://openlibrary.org/works/OL123456W"
    ) == "OL123456W"


def test_normalize_openlibrary_edition_id():
    assert normalize_openlibrary_id(
        "/books/OL123456M"
    ) == "OL123456M"


def test_title_author_key():
    assert build_title_author_key(
        "The Great Gatsby",
        "F. Scott Fitzgerald",
    ) == "the great gatsby||f scott fitzgerald"


def test_work_identity_takes_priority():
    identity = build_canonical_book_id(
        openlibrary_work_id="OL123456W",
        openlibrary_edition_id="OL987654M",
        isbn="9780000000000",
        title="Example Book",
        author="Example Author",
    )

    assert identity.identity_method == "openlibrary_work"
    assert identity.identity_confidence == "high"
    assert identity.openlibrary_work_id == "OL123456W"
    assert identity.openlibrary_edition_id == "OL987654M"


def test_edition_identity_fallback():
    identity = build_canonical_book_id(
        openlibrary_edition_id="OL987654M",
        title="Example Book",
        author="Example Author",
    )

    assert identity.identity_method == "openlibrary_edition"
    assert identity.identity_confidence == "high"


def test_isbn_identity_fallback():
    identity = build_canonical_book_id(
        isbn="978-0-06-112008-4",
        title="Example Book",
        author="Example Author",
    )

    assert identity.identity_method == "isbn"
    assert identity.identity_confidence == "medium"


def test_title_author_identity_fallback():
    identity = build_canonical_book_id(
        title="Example Book",
        author="Example Author",
    )

    assert identity.identity_method == "title_author"
    assert identity.identity_confidence == "low"


def test_source_fallback():
    identity = build_canonical_book_id(
        source="goodreads",
        source_book_id="review-123",
    )

    assert identity.identity_method == "source_fallback"
    assert identity.identity_confidence == "low"


def test_work_identity_is_deterministic():
    first = build_canonical_book_id(
        openlibrary_work_id="OL123456W",
    )

    second = build_canonical_book_id(
        openlibrary_work_id="OL123456W",
    )

    assert first.canonical_book_id == second.canonical_book_id


def test_same_work_produces_same_canonical_book():
    first = build_canonical_book_id(
        openlibrary_work_id="OL123456W",
        title="Example Book",
        author="Example Author",
    )

    second = build_canonical_book_id(
        openlibrary_work_id="OL123456W",
        title="Example Book",
        author="Different Edition Metadata",
    )

    assert first.canonical_book_id == second.canonical_book_id