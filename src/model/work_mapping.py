from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_BOOK_COLUMNS = {
    "canonical_book_id",
}

REQUIRED_AUDIT_COLUMNS = {
    "canonical_book_id_a",
    "canonical_book_id_b",
    "confidence",
}

VALID_CONFIDENCE_LEVELS = {
    "low",
    "medium",
    "high",
}

CONFIDENCE_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def validate_canonical_books(books: pd.DataFrame) -> None:
    """Validate the canonical book corpus."""
    missing = REQUIRED_BOOK_COLUMNS - set(books.columns)

    if missing:
        raise ValueError(
            "Canonical books are missing required columns: "
            + ", ".join(sorted(missing))
        )

    if books["canonical_book_id"].isna().any():
        raise ValueError(
            "Canonical books contain missing canonical_book_id values."
        )

    if books["canonical_book_id"].duplicated().any():
        raise ValueError(
            "Canonical books contain duplicate canonical_book_id values."
        )


def validate_reconciliation_audit(
    audit: pd.DataFrame,
) -> None:
    """
    Validate the reconciliation audit.

    An empty audit is valid as long as it still has the required
    columns. This represents a valid corpus with no confirmed
    reconciliation relationships.
    """
    missing = REQUIRED_AUDIT_COLUMNS - set(audit.columns)

    if missing:
        raise ValueError(
            "Reconciliation audit is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if audit.empty:
        return

    invalid_confidence = set(
        audit["confidence"].dropna().astype(str)
    ) - VALID_CONFIDENCE_LEVELS

    if invalid_confidence:
        raise ValueError(
            "Reconciliation audit contains invalid confidence values: "
            + ", ".join(sorted(invalid_confidence))
        )

    if audit[
        ["canonical_book_id_a", "canonical_book_id_b"]
    ].isna().any().any():
        raise ValueError(
            "Reconciliation audit contains missing book IDs."
        )

    if (
        audit["canonical_book_id_a"]
        == audit["canonical_book_id_b"]
    ).any():
        raise ValueError(
            "Reconciliation audit contains self-referential pairs."
        )


class UnionFind:
    """Simple deterministic union-find implementation."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {
            value: value
            for value in values
        }

    def find(self, value: str) -> str:
        """Find the root for a value with path compression."""
        if self.parent[value] != value:
            self.parent[value] = self.find(
                self.parent[value]
            )

        return self.parent[value]

    def union(
        self,
        first: str,
        second: str,
    ) -> None:
        """Union two values using deterministic root ordering."""
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if root_first < root_second:
            self.parent[root_second] = root_first
        else:
            self.parent[root_first] = root_second


def build_canonical_work_id(
    work_group_root: str,
) -> str:
    """
    Build a deterministic canonical work ID from the group root.
    """
    digest = hashlib.sha256(
        work_group_root.encode("utf-8")
    ).hexdigest()[:16]

    return f"work_{digest}"


def build_canonical_work_mapping(
    books: pd.DataFrame,
    reconciliation_audit: pd.DataFrame,
    minimum_confidence: str = "high",
) -> pd.DataFrame:
    """
    Build a canonical work mapping.

    High-confidence reconciliation relationships are treated as
    provisional same-work relationships.

    Medium- and low-confidence relationships remain separate unless
    the caller explicitly lowers the minimum confidence threshold.

    Every canonical book receives exactly one canonical_work_id.

    Original canonical book records are not modified or removed.
    """
    validate_canonical_books(books)
    validate_reconciliation_audit(
        reconciliation_audit
    )

    if minimum_confidence not in CONFIDENCE_RANK:
        raise ValueError(
            "minimum_confidence must be one of: "
            + ", ".join(
                sorted(CONFIDENCE_RANK)
            )
        )

    book_ids = (
        books["canonical_book_id"]
        .astype(str)
        .tolist()
    )

    book_id_set = set(book_ids)

    union_find = UnionFind(book_ids)

    eligible_pairs = reconciliation_audit[
        reconciliation_audit["confidence"].map(
            lambda value: CONFIDENCE_RANK.get(
                str(value),
                0,
            )
        )
        >= CONFIDENCE_RANK[minimum_confidence]
    ].copy()

    for _, row in eligible_pairs.iterrows():
        book_a = str(
            row["canonical_book_id_a"]
        )
        book_b = str(
            row["canonical_book_id_b"]
        )

        if book_a not in book_id_set:
            raise ValueError(
                f"Reconciliation audit references unknown "
                f"canonical book ID: {book_a}"
            )

        if book_b not in book_id_set:
            raise ValueError(
                f"Reconciliation audit references unknown "
                f"canonical book ID: {book_b}"
            )

        union_find.union(
            book_a,
            book_b,
        )

    roots = {
        book_id: union_find.find(book_id)
        for book_id in book_ids
    }

    group_sizes = pd.Series(
        list(roots.values())
    ).value_counts()

    rows = []

    for book_id in book_ids:
        root = roots[book_id]
        group_size = int(
            group_sizes[root]
        )

        if group_size > 1:
            status = "reconciled"
        else:
            status = "standalone"

        rows.append(
            {
                "canonical_book_id": book_id,
                "canonical_work_id": build_canonical_work_id(
                    root
                ),
                "work_mapping_status": status,
                "work_group_root": root,
            }
        )

    mapping = pd.DataFrame(rows)

    validate_work_mapping(
        mapping,
        books,
    )

    return mapping


def validate_work_mapping(
    mapping: pd.DataFrame,
    books: pd.DataFrame,
) -> None:
    """Validate the final canonical work mapping."""
    required_columns = {
        "canonical_book_id",
        "canonical_work_id",
        "work_mapping_status",
        "work_group_root",
    }

    missing = required_columns - set(
        mapping.columns
    )

    if missing:
        raise ValueError(
            "Work mapping is missing required columns: "
            + ", ".join(sorted(missing))
        )

    validate_canonical_books(books)

    expected_ids = set(
        books["canonical_book_id"]
        .astype(str)
    )

    actual_ids = set(
        mapping["canonical_book_id"]
        .astype(str)
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Work mapping must contain exactly one row "
            "for every canonical book."
        )

    if mapping[
        "canonical_book_id"
    ].duplicated().any():
        raise ValueError(
            "Work mapping contains duplicate canonical_book_id values."
        )

    if mapping[
        "canonical_work_id"
    ].isna().any():
        raise ValueError(
            "Work mapping contains missing canonical_work_id values."
        )

    valid_statuses = {
        "standalone",
        "reconciled",
    }

    invalid_statuses = set(
        mapping["work_mapping_status"]
        .astype(str)
    ) - valid_statuses

    if invalid_statuses:
        raise ValueError(
            "Invalid work_mapping_status values: "
            + ", ".join(sorted(invalid_statuses))
        )

    if mapping[
        "work_group_root"
    ].isna().any():
        raise ValueError(
            "Work mapping contains missing work_group_root values."
        )

    if not mapping[
        "canonical_work_id"
    ].astype(str).str.startswith("work_").all():
        raise ValueError(
            "All canonical_work_id values must start with 'work_'."
        )