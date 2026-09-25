from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


WORK_ID_PATTERN = re.compile(r"^OL\d+W$")
EDITION_ID_PATTERN = re.compile(r"^OL\d+M$")
AUTHOR_ID_PATTERN = re.compile(r"^OL\d+A$")


@dataclass(frozen=True)
class OpenLibraryIdentity:
    """
    Stable identity information for an Open Library book/work.

    work_id:
        Logical Open Library Work identifier, e.g. OL123W.

    edition_id:
        Specific Open Library Edition identifier, e.g. OL123M.

    matched_by:
        How the identity was established.
    """

    work_id: str | None
    edition_id: str | None
    matched_by: str | None


def normalize_openlibrary_key(value: Any) -> str | None:
    """Normalize an Open Library key or URL into its identifier."""
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    # Accept:
    # /works/OL123W
    # /books/OL123M
    # https://openlibrary.org/works/OL123W
    # OL123W
    match = re.search(r"/(?:works|books|authors)/(OL\d+[WMA])", text)

    if match:
        return match.group(1)

    if WORK_ID_PATTERN.fullmatch(text):
        return text

    if EDITION_ID_PATTERN.fullmatch(text):
        return text

    if AUTHOR_ID_PATTERN.fullmatch(text):
        return text

    return None


def normalize_isbn(value: Any) -> str | None:
    """Normalize an ISBN to digits, preserving X for ISBN-10."""
    if value is None:
        return None

    text = str(value).strip().upper()

    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    text = re.sub(r"[^0-9X]", "", text)

    return text or None


def normalize_title(value: Any) -> str:
    """Normalize a title for deterministic fallback matching."""
    if value is None:
        return ""

    text = str(value).strip().lower()

    if not text or text in {"nan", "none", "null"}:
        return ""

    text = unicodedata.normalize("NFKD", text)

    # Remove accents while retaining the underlying characters.
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_author(value: Any) -> str:
    """Normalize an author name for deterministic fallback matching."""
    return normalize_title(value)


def build_title_author_key(
    title: Any,
    author: Any,
) -> str | None:
    """Build a normalized title + author fallback key."""
    normalized_title = normalize_title(title)
    normalized_author = normalize_author(author)

    if not normalized_title or not normalized_author:
        return None

    return f"{normalized_title}||{normalized_author}"


def extract_work_id(
    *,
    work_key: Any = None,
    edition_work_key: Any = None,
    raw_work: dict[str, Any] | None = None,
) -> str | None:
    """
    Extract an Open Library Work ID from known enrichment fields.

    Priority:
        explicit work key
        edition's linked work key
        raw work record key
    """
    candidates = (
        work_key,
        edition_work_key,
        raw_work.get("key") if raw_work else None,
    )

    for candidate in candidates:
        normalized = normalize_openlibrary_key(candidate)

        if normalized and WORK_ID_PATTERN.fullmatch(normalized):
            return normalized

    return None


def extract_edition_id(
    *,
    edition_key: Any = None,
    raw_edition: dict[str, Any] | None = None,
) -> str | None:
    """Extract an Open Library Edition ID."""
    candidates = (
        edition_key,
        raw_edition.get("key") if raw_edition else None,
    )

    for candidate in candidates:
        normalized = normalize_openlibrary_key(candidate)

        if normalized and EDITION_ID_PATTERN.fullmatch(normalized):
            return normalized

    return None


def resolve_openlibrary_identity(
    *,
    work_key: Any = None,
    edition_key: Any = None,
    edition_work_key: Any = None,
    raw_work: dict[str, Any] | None = None,
    raw_edition: dict[str, Any] | None = None,
) -> OpenLibraryIdentity:
    """
    Resolve the strongest Open Library identity available.

    This function deliberately does not perform network requests.

    Network matching belongs in the ingestion/enrichment layer.
    """
    work_id = extract_work_id(
        work_key=work_key,
        edition_work_key=edition_work_key,
        raw_work=raw_work,
    )

    edition_id = extract_edition_id(
        edition_key=edition_key,
        raw_edition=raw_edition,
    )

    if work_id:
        matched_by = "openlibrary_work"

    elif edition_id:
        matched_by = "openlibrary_edition"

    else:
        matched_by = None

    return OpenLibraryIdentity(
        work_id=work_id,
        edition_id=edition_id,
        matched_by=matched_by,
    )