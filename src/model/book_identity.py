from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


OPENLIBRARY_WORK_PATTERN = re.compile(r"^OL\d+W$")
OPENLIBRARY_EDITION_PATTERN = re.compile(r"^OL\d+M$")


@dataclass(frozen=True)
class CanonicalBookIdentity:
    """
    Application-level identity for a book/work.

    canonical_book_id:
        Stable ID owned by this application.

    openlibrary_work_id:
        Open Library Work identifier when available.

    openlibrary_edition_id:
        Open Library Edition identifier when available.

    identity_method:
        How the identity was established.

    identity_confidence:
        Qualitative confidence in the identity.
    """

    canonical_book_id: str
    openlibrary_work_id: str | None
    openlibrary_edition_id: str | None
    identity_method: str
    identity_confidence: str


def _clean_value(value: Any) -> str:
    """Convert a value to a normalized string."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def normalize_title(value: Any) -> str:
    """
    Normalize a title for deterministic identity fallback.

    This is intentionally conservative. We are not trying to
    determine whether two books are the same based on title alone.
    """
    text = _clean_value(value).lower()

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_author(value: Any) -> str:
    """Normalize an author name for deterministic fallback identity."""
    return normalize_title(value)


def normalize_isbn(value: Any) -> str:
    """Normalize an ISBN while preserving ISBN-10 X."""
    text = _clean_value(value).upper()

    if not text:
        return ""

    return re.sub(r"[^0-9X]", "", text)


def normalize_openlibrary_id(value: Any) -> str | None:
    """
    Extract a normalized Open Library Work or Edition ID.

    Accepts values such as:

        OL123W
        OL123M
        /works/OL123W
        /books/OL123M
        https://openlibrary.org/works/OL123W
    """
    text = _clean_value(value)

    if not text:
        return None

    if OPENLIBRARY_WORK_PATTERN.fullmatch(text):
        return text

    if OPENLIBRARY_EDITION_PATTERN.fullmatch(text):
        return text

    match = re.search(
        r"/(?:works|books)/(OL\d+[WM])",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def build_title_author_key(
    title: Any,
    author: Any,
) -> str | None:
    """
    Build a deterministic title+author key.

    This is a fallback identity mechanism only.
    """
    normalized_title = normalize_title(title)
    normalized_author = normalize_author(author)

    if not normalized_title or not normalized_author:
        return None

    return f"{normalized_title}||{normalized_author}"


def build_canonical_book_id(
    *,
    openlibrary_work_id: str | None = None,
    openlibrary_edition_id: str | None = None,
    isbn: str | None = None,
    title: Any = None,
    author: Any = None,
    source: str | None = None,
    source_book_id: str | None = None,
) -> CanonicalBookIdentity:
    """
    Build an application-owned canonical book identity.

    Identity priority:

    1. Open Library Work
    2. Open Library Edition
    3. ISBN
    4. normalized title + author
    5. source + source_book_id

    The source-specific ID is deliberately the last fallback.

    The function never silently discards a book.
    """

    work_id = normalize_openlibrary_id(openlibrary_work_id)
    edition_id = normalize_openlibrary_id(openlibrary_edition_id)
    normalized_isbn = normalize_isbn(isbn)
    title_author_key = build_title_author_key(title, author)

    if work_id and OPENLIBRARY_WORK_PATTERN.fullmatch(work_id):
        canonical_key = f"openlibrary_work:{work_id}"

        return CanonicalBookIdentity(
            canonical_book_id=_hash_identity(canonical_key),
            openlibrary_work_id=work_id,
            openlibrary_edition_id=(
                edition_id
                if edition_id
                and OPENLIBRARY_EDITION_PATTERN.fullmatch(edition_id)
                else None
            ),
            identity_method="openlibrary_work",
            identity_confidence="high",
        )

    if (
        edition_id
        and OPENLIBRARY_EDITION_PATTERN.fullmatch(edition_id)
    ):
        canonical_key = f"openlibrary_edition:{edition_id}"

        return CanonicalBookIdentity(
            canonical_book_id=_hash_identity(canonical_key),
            openlibrary_work_id=None,
            openlibrary_edition_id=edition_id,
            identity_method="openlibrary_edition",
            identity_confidence="high",
        )

    if normalized_isbn:
        canonical_key = f"isbn:{normalized_isbn}"

        return CanonicalBookIdentity(
            canonical_book_id=_hash_identity(canonical_key),
            openlibrary_work_id=None,
            openlibrary_edition_id=None,
            identity_method="isbn",
            identity_confidence="medium",
        )

    if title_author_key:
        canonical_key = f"title_author:{title_author_key}"

        return CanonicalBookIdentity(
            canonical_book_id=_hash_identity(canonical_key),
            openlibrary_work_id=None,
            openlibrary_edition_id=None,
            identity_method="title_author",
            identity_confidence="low",
        )

    if source and source_book_id:
        canonical_key = (
            f"source:{source}:source_book_id:{source_book_id}"
        )

        return CanonicalBookIdentity(
            canonical_book_id=_hash_identity(canonical_key),
            openlibrary_work_id=None,
            openlibrary_edition_id=None,
            identity_method="source_fallback",
            identity_confidence="low",
        )

    raise ValueError(
        "Unable to construct a canonical book identity. "
        "At least one identity signal is required."
    )


def _hash_identity(value: str) -> str:
    """
    Hash an identity key into a compact deterministic ID.

    The prefix makes it clear that this is an application-level
    canonical identity rather than an external source ID.
    """
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]

    return f"book_{digest}"
