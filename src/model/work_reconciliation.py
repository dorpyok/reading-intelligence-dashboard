"""
Canonical work reconciliation utilities.

This module identifies pairs of canonical book records that may represent
the same underlying work.

Important:
    This module does NOT merge records.

It produces evidence that can be reviewed before establishing a
canonical-work mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


REQUIRED_COLUMNS = {
    "canonical_book_id",
    "title",
    "author",
    "isbn",
    "description",
    "openlibrary_work_id",
    "openlibrary_edition_id",
}


@dataclass(frozen=True)
class ReconciliationCandidate:
    """Evidence that two canonical records may represent the same work."""

    canonical_book_id_a: str
    canonical_book_id_b: str

    title_a: str
    title_b: str
    author_a: str
    author_b: str

    title_match: bool
    base_title_match: bool
    author_match: bool
    isbn_match: bool
    work_id_match: bool
    edition_id_match: bool

    description_similarity: float

    evidence_score: float
    confidence: str
    evidence_reasons: tuple[str, ...]


def normalize_text(value: object) -> str:
    """Normalize text while preserving substantive words."""
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def normalize_title(value: object) -> str:
    """
    Normalize a full title while preserving parenthetical information.
    """
    return normalize_text(value)


def normalize_base_title(value: object) -> str:
    """
    Normalize the primary title while removing trailing parenthetical
    series/number information.

    Example:
        The Ex Hex (The Ex Hex, #1)
        -> the ex hex
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Remove trailing parenthetical content BEFORE punctuation normalization.
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

    return normalize_text(text)


def normalize_isbn(value: object) -> str:
    """Normalize an ISBN to digits/X only."""
    if pd.isna(value):
        return ""

    return re.sub(r"[^0-9xX]", "", str(value)).upper()


def normalize_identifier(value: object) -> str:
    """Normalize an external identifier."""
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return ""

    return value.upper()


def tokenize(text: str) -> set[str]:
    """Return normalized word tokens."""
    return set(text.split())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculate token-level Jaccard similarity."""
    tokens_a = tokenize(normalize_text(text_a))
    tokens_b = tokenize(normalize_text(text_b))

    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def validate_canonical_books(dataframe: pd.DataFrame) -> None:
    """Validate the canonical corpus contains required fields."""
    missing = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Canonical book dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if dataframe["canonical_book_id"].isna().any():
        raise ValueError("canonical_book_id contains missing values.")

    if dataframe["canonical_book_id"].duplicated().any():
        raise ValueError("canonical_book_id contains duplicate values.")


def build_normalized_identity_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add normalized identity fields used during reconciliation."""
    validate_canonical_books(dataframe)

    result = dataframe.copy()

    result["normalized_title"] = result["title"].map(
        normalize_title
    )
    result["normalized_base_title"] = result["title"].map(
        normalize_base_title
    )
    result["normalized_author"] = result["author"].map(
        normalize_text
    )
    result["normalized_isbn"] = result["isbn"].map(
        normalize_isbn
    )
    result["normalized_work_id"] = result["openlibrary_work_id"].map(
        normalize_identifier
    )
    result["normalized_edition_id"] = result[
        "openlibrary_edition_id"
    ].map(normalize_identifier)

    return result


def _add_pair(
    pairs: set[tuple[str, str]],
    left_id: object,
    right_id: object,
) -> None:
    """Add an ordered, de-duplicated pair."""
    if pd.isna(left_id) or pd.isna(right_id):
        return

    left = str(left_id)
    right = str(right_id)

    if not left or not right or left == right:
        return

    pairs.add(tuple(sorted((left, right))))


def generate_candidate_pairs(
    dataframe: pd.DataFrame,
) -> list[tuple[str, str]]:
    """
    Generate conservative reconciliation candidates.

    Blocking uses:
        1. Exact normalized title + author
        2. Exact normalized base title + author
        3. Exact normalized title + work ID
        4. Exact normalized base title + work ID
        5. Exact ISBN
    """
    df = build_normalized_identity_columns(dataframe)

    pairs: set[tuple[str, str]] = set()

    block_columns = [
        ["normalized_title", "normalized_author"],
        ["normalized_base_title", "normalized_author"],
        ["normalized_title", "normalized_work_id"],
        ["normalized_base_title", "normalized_work_id"],
        ["normalized_isbn"],
    ]

    for columns in block_columns:
        valid = df.copy()

        for column in columns:
            valid = valid[valid[column] != ""]

        if valid.empty:
            continue

        for _, group in valid.groupby(columns, dropna=False):
            ids = group["canonical_book_id"].tolist()

            if len(ids) < 2:
                continue

            for i, left_id in enumerate(ids):
                for right_id in ids[i + 1 :]:
                    _add_pair(pairs, left_id, right_id)

    return sorted(pairs)


def _record_by_id(dataframe: pd.DataFrame) -> dict[str, pd.Series]:
    """Create a lookup dictionary keyed by canonical book ID."""
    return {
        str(row["canonical_book_id"]): row
        for _, row in dataframe.iterrows()
    }


def reconcile_pair(
    row_a: pd.Series,
    row_b: pd.Series,
) -> ReconciliationCandidate:
    """
    Evaluate identity evidence for one candidate pair.

    Confidence is intentionally conservative. Candidate generation and
    identity confirmation are separate decisions.
    """
    title_a = normalize_title(row_a["title"])
    title_b = normalize_title(row_b["title"])

    base_title_a = normalize_base_title(row_a["title"])
    base_title_b = normalize_base_title(row_b["title"])

    author_a = normalize_text(row_a["author"])
    author_b = normalize_text(row_b["author"])

    isbn_a = normalize_isbn(row_a["isbn"])
    isbn_b = normalize_isbn(row_b["isbn"])

    work_a = normalize_identifier(row_a["openlibrary_work_id"])
    work_b = normalize_identifier(row_b["openlibrary_work_id"])

    edition_a = normalize_identifier(row_a["openlibrary_edition_id"])
    edition_b = normalize_identifier(row_b["openlibrary_edition_id"])

    title_match = bool(title_a and title_a == title_b)
    base_title_match = bool(
        base_title_a and base_title_a == base_title_b
    )
    author_match = bool(author_a and author_a == author_b)
    isbn_match = bool(isbn_a and isbn_a == isbn_b)
    work_id_match = bool(work_a and work_a == work_b)
    edition_id_match = bool(
        edition_a and edition_a == edition_b
    )

    description_similarity = jaccard_similarity(
        row_a["description"],
        row_b["description"],
    )

    score = 0.0
    reasons: list[str] = []

    if work_id_match:
        score += 0.50
        reasons.append("same Open Library Work ID")

    if edition_id_match:
        score += 0.15
        reasons.append("same Open Library Edition ID")

    if isbn_match:
        score += 0.20
        reasons.append("same ISBN")

    if title_match:
        score += 0.15
        reasons.append("same normalized title")
    elif base_title_match:
        score += 0.10
        reasons.append("same normalized base title")

    if author_match:
        score += 0.10
        reasons.append("same normalized author")

    if description_similarity >= 0.90:
        score += 0.15
        reasons.append("very similar descriptions")
    elif description_similarity >= 0.75:
        score += 0.10
        reasons.append("similar descriptions")
    elif description_similarity >= 0.50:
        score += 0.05
        reasons.append("moderately similar descriptions")

    score = min(score, 1.0)

    # High confidence requires strong corroborating evidence.
    if work_id_match and (title_match or base_title_match):
        confidence = "high"
    elif isbn_match and (title_match or base_title_match) and author_match:
        confidence = "high"
    elif (
        base_title_match
        and author_match
        and description_similarity >= 0.90
    ):
        confidence = "high"
    elif (
        title_match
        and author_match
        and description_similarity >= 0.95
    ):
        confidence = "high"
    elif (
        base_title_match
        and author_match
        and description_similarity >= 0.75
    ):
        confidence = "medium"
    else:
        confidence = "low"

    return ReconciliationCandidate(
        canonical_book_id_a=str(row_a["canonical_book_id"]),
        canonical_book_id_b=str(row_b["canonical_book_id"]),
        title_a=str(row_a["title"]),
        title_b=str(row_b["title"]),
        author_a=str(row_a["author"]),
        author_b=str(row_b["author"]),
        title_match=title_match,
        base_title_match=base_title_match,
        author_match=author_match,
        isbn_match=isbn_match,
        work_id_match=work_id_match,
        edition_id_match=edition_id_match,
        description_similarity=description_similarity,
        evidence_score=score,
        confidence=confidence,
        evidence_reasons=tuple(reasons),
    )


def build_reconciliation_audit(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a reconciliation audit dataframe."""
    validate_canonical_books(dataframe)

    normalized = build_normalized_identity_columns(dataframe)
    lookup = _record_by_id(normalized)

    pairs = generate_candidate_pairs(normalized)

    candidates: list[dict[str, object]] = []

    for left_id, right_id in pairs:
        candidate = reconcile_pair(
            lookup[left_id],
            lookup[right_id],
        )

        candidates.append(
            {
                "canonical_book_id_a": candidate.canonical_book_id_a,
                "canonical_book_id_b": candidate.canonical_book_id_b,
                "title_a": candidate.title_a,
                "title_b": candidate.title_b,
                "author_a": candidate.author_a,
                "author_b": candidate.author_b,
                "title_match": candidate.title_match,
                "base_title_match": candidate.base_title_match,
                "author_match": candidate.author_match,
                "isbn_match": candidate.isbn_match,
                "work_id_match": candidate.work_id_match,
                "edition_id_match": candidate.edition_id_match,
                "description_similarity": candidate.description_similarity,
                "evidence_score": candidate.evidence_score,
                "confidence": candidate.confidence,
                "evidence_reasons": " | ".join(
                    candidate.evidence_reasons
                ),
            }
        )

    columns = [
        "canonical_book_id_a",
        "canonical_book_id_b",
        "title_a",
        "title_b",
        "author_a",
        "author_b",
        "title_match",
        "base_title_match",
        "author_match",
        "isbn_match",
        "work_id_match",
        "edition_id_match",
        "description_similarity",
        "evidence_score",
        "confidence",
        "evidence_reasons",
    ]

    if not candidates:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(candidates, columns=columns)
        .sort_values(
            ["evidence_score", "description_similarity"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_work_groups_from_confirmed_pairs(
    audit: pd.DataFrame,
    minimum_confidence: str = "high",
) -> dict[str, str]:
    """
    Build provisional work groups from reviewed reconciliation pairs.

    This should only be used after review of the audit output.
    """
    confidence_rank = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    if minimum_confidence not in confidence_rank:
        raise ValueError(
            "minimum_confidence must be one of: low, medium, high"
        )

    if audit.empty:
        return {}

    required_columns = {
        "canonical_book_id_a",
        "canonical_book_id_b",
        "confidence",
    }

    missing = required_columns - set(audit.columns)

    if missing:
        raise ValueError(
            "Audit dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)

        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]

        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)

        if left_root != right_root:
            parent[right_root] = left_root

    for _, row in audit.iterrows():
        confidence = str(row["confidence"])

        if confidence_rank.get(confidence, -1) < confidence_rank[
            minimum_confidence
        ]:
            continue

        union(
            str(row["canonical_book_id_a"]),
            str(row["canonical_book_id_b"]),
        )

    groups: dict[str, str] = {}

    for node in parent:
        groups[node] = find(node)

    return groups