from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://openlibrary.org"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "openlibrary_cache"

REQUEST_DELAY_SECONDS = 0.2
SEARCH_LIMIT = 100


@dataclass
class OpenLibraryEnrichment:
    """
    Metadata enrichment for a canonical Book.

    This object describes the book itself.
    It does not contain reader-specific reading status,
    ratings, shelves, or dates.
    """

    source: str = "openlibrary"

    openlibrary_work_id: str | None = None
    openlibrary_edition_id: str | None = None

    matched_by: str | None = None
    match_score: float | None = None

    title: str | None = None
    authors: list[str] | None = None
    publication_year: int | None = None

    subjects: list[str] | None = None
    subject_people: list[str] | None = None
    subject_places: list[str] | None = None
    subject_times: list[str] | None = None

    isbn_10: list[str] | None = None
    isbn_13: list[str] | None = None

    cover_url: str | None = None

    raw_work: dict[str, Any] | None = None
    raw_edition: dict[str, Any] | None = None

    status: str = "unmatched"


def normalize_title(title: str) -> str:
    """
    Normalize a title for matching.

    Handles:
    - case
    - punctuation
    - subtitles
    - trailing series information in parentheses
    - common 'Title, The' ordering
    """

    if not title:
        return ""

    value = title.strip().lower()

    # Remove trailing parenthetical series information.
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)

    # Remove subtitle.
    value = value.split(":", 1)[0]

    # Normalize "Title, The" -> "The Title"
    match = re.match(r"^(.*),\s*(the|a|an)$", value)

    if match:
        value = f"{match.group(2)} {match.group(1)}"

    # Remove punctuation.
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_author(author: str) -> str:
    """
    Normalize an author name for matching.
    """

    if not author:
        return ""

    value = author.lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def similarity(left: str, right: str) -> float:
    """
    Return a normalized similarity score between 0 and 1.
    """

    if not left or not right:
        return 0.0

    return SequenceMatcher(
        None,
        left,
        right,
    ).ratio()


def resolve_author_name(
    client: "OpenLibraryClient",
    author_key: str,
) -> str | None:
    """
    Resolve an Open Library author key to a human-readable name.
    """

    if not author_key:
        return None

    author_id = author_key.split("/")[-1]

    author = client._get_json(
        f"{BASE_URL}/authors/{author_id}.json"
    )

    if not author:
        return None

    return author.get("name")


def extract_authors(
    client: "OpenLibraryClient",
    record: dict[str, Any],
) -> list[str]:
    """
    Extract human-readable author names from an Open Library
    Work or Edition record.
    """

    authors = []

    for author in record.get("authors", []):
        if not isinstance(author, dict):
            continue

        # Some Open Library responses may already contain a name.
        name = author.get("name")

        if name:
            authors.append(name)
            continue

        # Work records commonly use:
        # {"author": {"key": "/authors/OL..."}}
        author_data = author.get("author", {})

        author_key = author_data.get("key")

        # Edition records may use:
        # {"key": "/authors/OL..."}
        if not author_key:
            author_key = author.get("key")

        if author_key:
            resolved_name = resolve_author_name(
                client,
                author_key,
            )

            if resolved_name:
                authors.append(resolved_name)

    return sorted(set(authors))


def extract_subject_names(
    values: list[Any] | None,
) -> list[str]:
    """
    Normalize Open Library subject values.

    Open Library may return either strings or dictionaries.
    """

    if not values:
        return []

    results = []

    for value in values:
        if isinstance(value, str):
            results.append(value.strip())

        elif isinstance(value, dict):
            name = value.get("name")

            if name:
                results.append(
                    str(name).strip()
                )

    return sorted(
        set(
            x
            for x in results
            if x
        )
    )


def extract_year(
    record: dict[str, Any],
    edition: dict[str, Any] | None = None,
) -> int | None:
    """
    Extract a publication year from the Work or Edition.
    """

    # First preference: Work-level first publication year.
    year = record.get("first_publish_year")

    if year:
        try:
            return int(year)
        except (TypeError, ValueError):
            pass

    # Second preference: Edition publication date.
    if edition:
        publish_date = edition.get("publish_date")

        if publish_date:
            match = re.search(
                r"\b(1[0-9]{3}|20[0-9]{2})\b",
                str(publish_date),
            )

            if match:
                return int(
                    match.group(1)
                )

    # Final fallback: Work-level publication date.
    publish_date = record.get("publish_date")

    if publish_date:
        match = re.search(
            r"\b(1[0-9]{3}|20[0-9]{2})\b",
            str(publish_date),
        )

        if match:
            return int(
                match.group(1)
            )

    return None


class OpenLibraryClient:
    """
    Small production client for Open Library metadata.

    Responsibilities:
    - HTTP requests
    - caching
    - book matching
    - Work/Edition retrieval
    - metadata extraction

    It does not perform reader analytics.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
    ) -> None:

        self.cache_dir = cache_dir

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.request_delay = request_delay

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "ReadingIntelligenceDashboard/0.1 "
                    "(metadata enrichment project)"
                )
            }
        )

    def _cache_path(
        self,
        url: str,
    ) -> Path:

        safe_name = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            url,
        )

        return (
            self.cache_dir
            / f"{safe_name}.json"
        )

    def _get_json(
        self,
        url: str,
    ) -> dict[str, Any] | None:

        cache_path = self._cache_path(url)

        # Use cached response when available.
        if cache_path.exists():
            try:
                return json.loads(
                    cache_path.read_text(
                        encoding="utf-8"
                    )
                )

            except json.JSONDecodeError:
                cache_path.unlink()

        time.sleep(
            self.request_delay
        )

        response = self.session.get(
            url,
            timeout=30,
        )

        if response.status_code != 200:
            return None

        data = response.json()

        cache_path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return data

    def get_work(
        self,
        work_id: str,
    ) -> dict[str, Any] | None:

        return self._get_json(
            f"{BASE_URL}/works/{work_id}.json"
        )

    def get_edition(
        self,
        edition_id: str,
    ) -> dict[str, Any] | None:

        return self._get_json(
            f"{BASE_URL}/books/{edition_id}.json"
        )

    def lookup_isbn(
        self,
        isbn: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ] | None:

        if not isbn:
            return None

        clean_isbn = re.sub(
            r"[^0-9Xx]",
            "",
            str(isbn),
        )

        if not clean_isbn:
            return None

        edition = self._get_json(
            f"{BASE_URL}/isbn/{clean_isbn}.json"
        )

        if not edition:
            return None

        edition_key = edition.get(
            "key",
            "",
        )

        if not edition_key:
            return None

        edition_id = edition_key.split(
            "/"
        )[-1]

        work_keys = edition.get(
            "works",
            [],
        )

        if not work_keys:
            return None

        work_key = work_keys[0].get(
            "key"
        )

        if not work_key:
            return None

        work_id = work_key.split(
            "/"
        )[-1]

        work = self.get_work(
            work_id
        )

        if not work:
            return None

        return work, edition

    def search_title_author(
        self,
        title: str,
        author: str,
    ) -> list[dict[str, Any]]:

        params = {
            "title": title,
            "author": author,
            "limit": SEARCH_LIMIT,
        }

        return self._search(
            params
        )

    def search_author(
        self,
        author: str,
    ) -> list[dict[str, Any]]:

        params = {
            "author": author,
            "limit": SEARCH_LIMIT,
        }

        return self._search(
            params
        )

    def _search(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:

        query = "&".join(
            f"{key}="
            f"{requests.utils.quote(str(value))}"
            for key, value in params.items()
        )

        url = (
            f"{BASE_URL}/search.json?"
            f"{query}"
        )

        data = self._get_json(
            url
        )

        if not data:
            return []

        return data.get(
            "docs",
            []
        )

    def match_book(
        self,
        title: str,
        author: str,
        isbn: str | None = None,
    ) -> OpenLibraryEnrichment:

        # ---------------------------------------------------------
        # 1. ISBN
        # ---------------------------------------------------------

        if isbn:
            isbn_result = self.lookup_isbn(
                isbn
            )

            if isbn_result:
                work, edition = isbn_result

                work_id = work.get(
                    "key",
                    "",
                ).split("/")[-1]

                edition_id = edition.get(
                    "key",
                    "",
                ).split("/")[-1]

                return self._build_enrichment(
                    work=work,
                    edition=edition,
                    matched_by="isbn",
                    match_score=1.0,
                    work_id=work_id,
                    edition_id=edition_id,
                )

        normalized_title = normalize_title(
            title
        )

        normalized_author = normalize_author(
            author
        )

        # ---------------------------------------------------------
        # 2. Title + author search
        # ---------------------------------------------------------

        candidates = self.search_title_author(
            title=title,
            author=author,
        )

        best_candidate = self._find_best_candidate(
            candidates=candidates,
            title=normalized_title,
            author=normalized_author,
            minimum_score=0.85,
        )

        if best_candidate:

            return self._candidate_to_enrichment(
                best_candidate,
                matched_by="author_fuzzy_title",
                score=self._candidate_score(
                    best_candidate,
                    normalized_title,
                    normalized_author,
                ),
            )

        # ---------------------------------------------------------
        # 3. Author-first fallback
        # ---------------------------------------------------------

        author_candidates = self.search_author(
            author
        )

        best_candidate = self._find_best_candidate(
            candidates=author_candidates,
            title=normalized_title,
            author=normalized_author,
            minimum_score=0.82,
        )

        if best_candidate:

            return self._candidate_to_enrichment(
                best_candidate,
                matched_by="author_scan_title",
                score=self._candidate_score(
                    best_candidate,
                    normalized_title,
                    normalized_author,
                ),
            )

        # ---------------------------------------------------------
        # 4. No reliable match
        # ---------------------------------------------------------

        return OpenLibraryEnrichment(
            status="unmatched"
        )

    def _candidate_score(
        self,
        candidate: dict[str, Any],
        target_title: str,
        target_author: str,
    ) -> float:

        candidate_title = normalize_title(
            candidate.get(
                "title",
                "",
            )
        )

        candidate_authors = candidate.get(
            "author_name",
            [],
        )

        if isinstance(
            candidate_authors,
            str,
        ):
            candidate_authors = [
                candidate_authors
            ]

        candidate_author = (
            normalize_author(
                candidate_authors[0]
            )
            if candidate_authors
            else ""
        )

        title_score = similarity(
            target_title,
            candidate_title,
        )

        author_score = similarity(
            target_author,
            candidate_author,
        )

        return (
            title_score * 0.7
        ) + (
            author_score * 0.3
        )

    def _find_best_candidate(
        self,
        candidates: list[dict[str, Any]],
        title: str,
        author: str,
        minimum_score: float,
    ) -> dict[str, Any] | None:

        best = None
        best_score = 0.0

        for candidate in candidates:

            candidate_authors = candidate.get(
                "author_name",
                [],
            )

            if isinstance(
                candidate_authors,
                str,
            ):
                candidate_authors = [
                    candidate_authors
                ]

            candidate_author = (
                normalize_author(
                    candidate_authors[0]
                )
                if candidate_authors
                else ""
            )

            author_score = similarity(
                author,
                candidate_author,
            )

            # Never accept a title match with
            # a clearly different author.
            if author_score < 0.85:
                continue

            score = self._candidate_score(
                candidate,
                title,
                author,
            )

            if (
                score >= minimum_score
                and score > best_score
            ):
                best = candidate
                best_score = score

        return best

    def _candidate_to_enrichment(
        self,
        candidate: dict[str, Any],
        matched_by: str,
        score: float,
    ) -> OpenLibraryEnrichment:

        key = candidate.get(
            "key",
            "",
        )

        if not key:
            return OpenLibraryEnrichment(
                status="unmatched"
            )

        work_id = key.split(
            "/"
        )[-1]

        work = self.get_work(
            work_id
        )

        if not work:
            return OpenLibraryEnrichment(
                matched_by=matched_by,
                match_score=score,
                openlibrary_work_id=work_id,
                status="matched_work_fetch_failed",
            )

        return self._build_enrichment(
            work=work,
            edition=None,
            matched_by=matched_by,
            match_score=score,
            work_id=work_id,
            edition_id=None,
        )

    def _build_enrichment(
        self,
        work: dict[str, Any],
        edition: dict[str, Any] | None,
        matched_by: str,
        match_score: float,
        work_id: str | None,
        edition_id: str | None,
    ) -> OpenLibraryEnrichment:

        subjects = extract_subject_names(
            work.get(
                "subjects"
            )
        )

        subject_people = extract_subject_names(
            work.get(
                "subject_people"
            )
        )

        subject_places = extract_subject_names(
            work.get(
                "subject_places"
            )
        )

        subject_times = extract_subject_names(
            work.get(
                "subject_times"
            )
        )

        # Add edition-level subjects when available.
        if edition:
            subjects.extend(
                extract_subject_names(
                    edition.get(
                        "subjects"
                    )
                )
            )

        subjects = sorted(
            set(subjects)
        )

        # Resolve author IDs into names.
        authors = extract_authors(
            self,
            work,
        )

        isbn_10 = []
        isbn_13 = []

        if edition:

            isbn_10 = [
                str(value)
                for value in edition.get(
                    "isbn_10",
                    [],
                )
            ]

            isbn_13 = [
                str(value)
                for value in edition.get(
                    "isbn_13",
                    [],
                )
            ]

        # Build cover URL.
        cover_id = work.get(
            "covers",
            [None],
        )[0]

        cover_url = None

        if cover_id:
            cover_url = (
                f"{BASE_URL.replace('https://openlibrary.org', 'https://covers.openlibrary.org')}"
                f"/b/id/{cover_id}-L.jpg"
            )

        return OpenLibraryEnrichment(
            openlibrary_work_id=work_id,
            openlibrary_edition_id=edition_id,
            matched_by=matched_by,
            match_score=match_score,
            title=work.get(
                "title"
            ),
            authors=authors,
            publication_year=extract_year(
                work,
                edition,
            ),
            subjects=subjects,
            subject_people=subject_people,
            subject_places=subject_places,
            subject_times=subject_times,
            isbn_10=isbn_10,
            isbn_13=isbn_13,
            cover_url=cover_url,
            raw_work=work,
            raw_edition=edition,
            status="matched",
        )


def enrich_book(
    title: str,
    author: str,
    isbn: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function for enriching a single book.
    """

    client = OpenLibraryClient()

    enrichment = client.match_book(
        title=title,
        author=author,
        isbn=isbn,
    )

    return asdict(
        enrichment
    )