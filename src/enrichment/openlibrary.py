from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openlibrary_cache"
)


@dataclass
class OpenLibraryEnrichment:
    source: str = "openlibrary"

    openlibrary_work_id: str | None = None
    openlibrary_edition_id: str | None = None

    matched_by: str | None = None
    match_score: float | None = None

    title: str | None = None
    authors: list[str] = field(default_factory=list)

    publication_year: int | None = None

    subjects: list[str] = field(default_factory=list)
    subject_people: list[str] = field(default_factory=list)
    subject_places: list[str] = field(default_factory=list)
    subject_times: list[str] = field(default_factory=list)

    isbn_10: str | None = None
    isbn_13: str | None = None

    cover_url: str | None = None

    raw_work: dict[str, Any] | None = None
    raw_edition: dict[str, Any] | None = None

    status: str = "unmatched"


class OpenLibraryClient:
    BASE_URL = "https://openlibrary.org"

    TITLE_THRESHOLD = 0.85
    AUTHOR_THRESHOLD = 0.85
    MATCH_THRESHOLD = 0.82

    SEARCH_LIMIT = 100

    def __init__(
        self,
        cache_dir: Path | None = None,
        request_delay: float = 0.1,
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.request_delay = request_delay

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "ReadingIntelligenceDashboard/1.0 "
                    "(educational portfolio project)"
                )
            }
        )

    # ================================================================
    # CACHE
    # ================================================================

    def _cache_path(self, url: str) -> Path:
        """
        Create a filesystem-safe cache filename.

        The URL itself is not used as the filename.
        A SHA-256 hash prevents Windows path-length and
        invalid-character problems.
        """
        url_hash = hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()

        return self.cache_dir / f"{url_hash}.json"

    def _read_cache(self, cache_path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(
                cache_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None

    def _write_cache(
        self,
        cache_path: Path,
        data: dict[str, Any],
    ) -> None:
        try:
            cache_path.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                "Warning: could not cache "
                "Open Library response:"
            )
            print(f"  {exc}")

    def _get_json(
        self,
        url: str,
    ) -> dict[str, Any] | None:
        cache_path = self._cache_path(url)

        cached = self._read_cache(cache_path)

        if cached is not None:
            return cached

        time.sleep(self.request_delay)

        try:
            response = self.session.get(
                url,
                timeout=30,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as exc:
            print("Open Library request failed:")
            print(f"  URL: {url}")
            print(f"  Error: {exc}")
            return None

        if isinstance(data, dict):
            self._write_cache(
                cache_path,
                data,
            )

        return data

    def _search_cache_url(
        self,
        params: dict[str, Any],
    ) -> str:
        """
        Create a deterministic cache key for a search request.

        Search responses were previously not cached, which meant
        repeated runs and repeated books could make the same
        Open Library search request again.
        """
        query = urlencode(
            sorted(
                (
                    str(key),
                    str(value),
                )
                for key, value in params.items()
            )
        )

        return f"{self.BASE_URL}/search.json?{query}"

    # ================================================================
    # NORMALIZATION
    # ================================================================

    @staticmethod
    def normalize_text(
        value: str | None,
    ) -> str:
        if not value:
            return ""

        value = value.lower()

        value = re.sub(
            r"\([^)]*\)",
            "",
            value,
        )

        value = re.sub(
            r"[^a-z0-9]+",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    @classmethod
    def normalize_title(
        cls,
        title: str | None,
    ) -> str:
        return cls.normalize_text(title)

    @classmethod
    def normalize_author(
        cls,
        author: str | None,
    ) -> str:
        return cls.normalize_text(author)

    # ================================================================
    # SIMILARITY
    # ================================================================

    @staticmethod
    def similarity(
        left: str,
        right: str,
    ) -> float:
        if not left or not right:
            return 0.0

        return SequenceMatcher(
            None,
            left,
            right,
        ).ratio()

    # ================================================================
    # ISBN
    # ================================================================

    def lookup_isbn(
        self,
        isbn: str,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        isbn = str(isbn).strip()

        if not isbn:
            return None, None

        url = (
            f"{self.BASE_URL}"
            f"/isbn/{isbn}.json"
        )

        edition = self._get_json(url)

        if not edition:
            return None, None

        work_keys = edition.get(
            "works",
            [],
        )

        if not work_keys:
            return edition, None

        first_work = work_keys[0]

        if isinstance(first_work, dict):
            work_key = first_work.get("key")
        else:
            work_key = first_work

        if not work_key:
            return edition, None

        if work_key.startswith("/works/"):
            work_url = (
                f"{self.BASE_URL}"
                f"{work_key}.json"
            )
        else:
            work_url = (
                f"{self.BASE_URL}"
                f"/works/{work_key}.json"
            )

        work = self._get_json(work_url)

        return edition, work

    # ================================================================
    # SEARCH
    # ================================================================

    def _search(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute and cache an Open Library search request.

        This is the major performance improvement over the previous
        implementation: search responses now use the same filesystem
        cache as work/edition/author requests.
        """
        cache_url = self._search_cache_url(params)
        cache_path = self._cache_path(cache_url)

        cached = self._read_cache(cache_path)

        if cached is not None:
            return cached.get("docs", [])

        time.sleep(self.request_delay)

        try:
            response = self.session.get(
                f"{self.BASE_URL}/search.json",
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as exc:
            print("Open Library search failed:")
            print(f"  Parameters: {params}")
            print(f"  Error: {exc}")
            return []

        if isinstance(data, dict):
            self._write_cache(
                cache_path,
                data,
            )

        return data.get("docs", [])

    def search_books(
        self,
        title: str,
        author: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "title": title,
            "limit": self.SEARCH_LIMIT,
        }

        if author:
            params["author"] = author

        return self._search(params)

    def search_title_author(
        self,
        title: str,
        author: str,
    ) -> list[dict[str, Any]]:
        return self.search_books(
            title=title,
            author=author,
        )

    def search_author(
        self,
        author: str,
    ) -> list[dict[str, Any]]:
        params = {
            "author": author,
            "limit": self.SEARCH_LIMIT,
        }

        return self._search(params)

    # ================================================================
    # AUTHOR
    # ================================================================

    def resolve_author_name(
        self,
        author_key: str,
    ) -> str | None:
        if not author_key:
            return None

        if author_key.startswith("/authors/"):
            url = (
                f"{self.BASE_URL}"
                f"{author_key}.json"
            )
        else:
            url = (
                f"{self.BASE_URL}"
                f"/authors/{author_key}.json"
            )

        data = self._get_json(url)

        if not data:
            return None

        return data.get("name")

    # ================================================================
    # WORK
    # ================================================================

    def get_work(
        self,
        work_key: str,
    ) -> dict[str, Any] | None:
        if not work_key:
            return None

        if work_key.startswith("/works/"):
            url = (
                f"{self.BASE_URL}"
                f"{work_key}.json"
            )
        else:
            url = (
                f"{self.BASE_URL}"
                f"/works/{work_key}.json"
            )

        return self._get_json(url)

    # ================================================================
    # SUBJECTS
    # ================================================================

    @staticmethod
    def extract_subjects(
        work: dict[str, Any] | None,
        edition: dict[str, Any] | None = None,
    ) -> list[str]:
        subjects = []

        if work:
            for subject in work.get(
                "subjects",
                [],
            ):
                if isinstance(subject, str):
                    subjects.append(subject)
                elif isinstance(subject, dict):
                    name = subject.get("name")
                    if name:
                        subjects.append(str(name))

        if edition:
            for subject in edition.get(
                "subjects",
                [],
            ):
                if isinstance(subject, str):
                    subjects.append(subject)
                elif isinstance(subject, dict):
                    name = subject.get("name")
                    if name:
                        subjects.append(str(name))

        cleaned = []
        seen = set()

        for subject in subjects:
            subject = str(subject).strip()

            if not subject:
                continue

            normalized = subject.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(subject)

        return cleaned

    @staticmethod
    def extract_subject_category(
        work: dict[str, Any] | None,
        field_name: str,
    ) -> list[str]:
        if not work:
            return []

        values = work.get(
            field_name,
            [],
        )

        results = []

        for value in values:
            if isinstance(value, str):
                results.append(value)

            elif isinstance(value, dict):
                name = value.get("name")

                if name:
                    results.append(str(name))

        return list(
            dict.fromkeys(
                value.strip()
                for value in results
                if value.strip()
            )
        )

    # ================================================================
    # PUBLICATION YEAR
    # ================================================================

    @staticmethod
    def extract_publication_year(
        work: dict[str, Any] | None,
        edition: dict[str, Any] | None,
    ) -> int | None:
        if work:
            year = work.get(
                "first_publish_year"
            )

            if year:
                try:
                    return int(year)
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        if edition:
            publish_date = edition.get(
                "publish_date"
            )

            if publish_date:
                match = re.search(
                    r"\b(1[5-9]\d{2}|20\d{2})\b",
                    str(publish_date),
                )

                if match:
                    return int(match.group(1))

        return None

    # ================================================================
    # ISBN EXTRACTION
    # ================================================================

    @staticmethod
    def extract_isbns(
        edition: dict[str, Any] | None,
    ) -> tuple[
        str | None,
        str | None,
    ]:
        if not edition:
            return None, None

        isbn_10_values = edition.get(
            "isbn_10",
            [],
        )

        isbn_13_values = edition.get(
            "isbn_13",
            [],
        )

        isbn_10 = (
            str(isbn_10_values[0])
            if isbn_10_values
            else None
        )

        isbn_13 = (
            str(isbn_13_values[0])
            if isbn_13_values
            else None
        )

        return isbn_10, isbn_13

    # ================================================================
    # COVER
    # ================================================================

    @staticmethod
    def extract_cover_url(
        edition: dict[str, Any] | None,
    ) -> str | None:
        if not edition:
            return None

        covers = edition.get(
            "covers",
            [],
        )

        if not covers:
            return None

        cover_id = covers[0]

        return (
            "https://covers.openlibrary.org/"
            f"b/id/{cover_id}-L.jpg"
        )

    # ================================================================
    # MATCHING
    # ================================================================

    def _candidate_score(
        self,
        candidate: dict[str, Any],
        target_title: str,
        target_author: str,
    ) -> float:
        candidate_title = self.normalize_title(
            candidate.get("title", "")
        )

        candidate_authors = candidate.get(
            "author_name",
            [],
        )

        if isinstance(candidate_authors, str):
            candidate_authors = [candidate_authors]

        author_scores = [
            self.similarity(
                target_author,
                self.normalize_author(candidate_author),
            )
            for candidate_author in candidate_authors
        ]

        author_score = max(
            author_scores,
            default=0.0,
        )

        title_score = self.similarity(
            target_title,
            candidate_title,
        )

        return (
            title_score * 0.7
            + author_score * 0.3
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

            if isinstance(candidate_authors, str):
                candidate_authors = [candidate_authors]

            author_scores = [
                self.similarity(
                    author,
                    self.normalize_author(candidate_author),
                )
                for candidate_author in candidate_authors
            ]

            author_score = max(
                author_scores,
                default=0.0,
            )

            if author_score < self.AUTHOR_THRESHOLD:
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

    def match_search_result(
        self,
        title: str,
        author: str,
        results: list[dict[str, Any]],
        minimum_score: float | None = None,
    ) -> tuple[
        dict[str, Any] | None,
        float,
    ]:
        normalized_title = self.normalize_title(title)
        normalized_author = self.normalize_author(author)

        minimum_score = (
            minimum_score
            if minimum_score is not None
            else self.TITLE_THRESHOLD
        )

        best_result = self._find_best_candidate(
            candidates=results,
            title=normalized_title,
            author=normalized_author,
            minimum_score=minimum_score,
        )

        if best_result is None:
            return None, 0.0

        score = self._candidate_score(
            best_result,
            normalized_title,
            normalized_author,
        )

        return best_result, score

    # ================================================================
    # SEARCH RESULT → WORK
    # ================================================================

    def enrich_search_result(
        self,
        result: dict[str, Any],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        work_key = result.get("key")

        if not work_key:
            return None, None

        work = self.get_work(work_key)

        if not work:
            return None, None

        edition = None

        edition_keys = result.get(
            "edition_key",
            [],
        )

        if edition_keys:
            edition_key = edition_keys[0]

            edition_url = (
                f"{self.BASE_URL}"
                f"/books/{edition_key}.json"
            )

            edition = self._get_json(
                edition_url
            )

        return work, edition

    # ================================================================
    # MATCH BOOK
    # ================================================================

    def match_book(
        self,
        title: str,
        author: str,
        isbn: str | None = None,
    ) -> OpenLibraryEnrichment:
        result = OpenLibraryEnrichment(
            title=title
        )

        normalized_title = self.normalize_title(title)
        normalized_author = self.normalize_author(author)

        # ------------------------------------------------------------
        # 1. ISBN EXACT MATCH
        # ------------------------------------------------------------

        if isbn:
            isbn_clean = str(isbn).strip()

            if isbn_clean:
                edition, work = self.lookup_isbn(
                    isbn_clean
                )

                if work:
                    result.status = "matched"
                    result.matched_by = "isbn"
                    result.match_score = 1.0

                    self.populate_result(
                        result,
                        work,
                        edition,
                    )

                    return result

        # ------------------------------------------------------------
        # 2. TITLE + AUTHOR FUZZY MATCH
        # ------------------------------------------------------------

        candidates = self.search_title_author(
            title=title,
            author=author,
        )

        best_candidate = self._find_best_candidate(
            candidates=candidates,
            title=normalized_title,
            author=normalized_author,
            minimum_score=self.TITLE_THRESHOLD,
        )

        if best_candidate:
            score = self._candidate_score(
                best_candidate,
                normalized_title,
                normalized_author,
            )

            work, edition = self.enrich_search_result(
                best_candidate
            )

            if work:
                result.status = "matched"
                result.matched_by = "author_fuzzy_title"
                result.match_score = score

                self.populate_result(
                    result,
                    work,
                    edition,
                )

                return result

        # ------------------------------------------------------------
        # 3. AUTHOR-FIRST FALLBACK
        # ------------------------------------------------------------

        if author:
            author_candidates = self.search_author(
                author
            )

            best_candidate = self._find_best_candidate(
                candidates=author_candidates,
                title=normalized_title,
                author=normalized_author,
                minimum_score=self.MATCH_THRESHOLD,
            )

            if best_candidate:
                score = self._candidate_score(
                    best_candidate,
                    normalized_title,
                    normalized_author,
                )

                work, edition = self.enrich_search_result(
                    best_candidate
                )

                if work:
                    result.status = "matched"
                    result.matched_by = "author_scan_title"
                    result.match_score = score

                    self.populate_result(
                        result,
                        work,
                        edition,
                    )

                    return result

        # ------------------------------------------------------------
        # 4. NO RELIABLE MATCH
        # ------------------------------------------------------------

        result.status = "unmatched"

        return result

    # ================================================================
    # POPULATE RESULT
    # ================================================================

    def populate_result(
        self,
        result: OpenLibraryEnrichment,
        work: dict[str, Any],
        edition: dict[str, Any] | None,
    ) -> None:
        result.raw_work = work
        result.raw_edition = edition

        work_key = work.get("key")

        if work_key:
            result.openlibrary_work_id = (
                work_key.split("/")[-1]
            )

        if edition:
            edition_key = edition.get("key")

            if edition_key:
                result.openlibrary_edition_id = (
                    edition_key.split("/")[-1]
                )

        result.title = work.get("title")

        if not result.title and edition:
            result.title = edition.get("title")

        authors = []

        for author_entry in work.get(
            "authors",
            [],
        ):
            if not isinstance(
                author_entry,
                dict,
            ):
                continue

            author_key = author_entry.get(
                "author",
                {},
            )

            if isinstance(
                author_key,
                dict,
            ):
                author_key = author_key.get("key")

            if not author_key:
                continue

            author_name = self.resolve_author_name(
                author_key
            )

            if author_name:
                authors.append(author_name)

        result.authors = list(
            dict.fromkeys(authors)
        )

        result.publication_year = (
            self.extract_publication_year(
                work,
                edition,
            )
        )

        result.subjects = self.extract_subjects(
            work,
            edition,
        )

        result.subject_people = (
            self.extract_subject_category(
                work,
                "subject_people",
            )
        )

        result.subject_places = (
            self.extract_subject_category(
                work,
                "subject_places",
            )
        )

        result.subject_times = (
            self.extract_subject_category(
                work,
                "subject_times",
            )
        )

        (
            result.isbn_10,
            result.isbn_13,
        ) = self.extract_isbns(
            edition
        )

        result.cover_url = self.extract_cover_url(
            edition
        )


def enrich_book(
    title: str,
    author: str,
    isbn: str | None = None,
) -> OpenLibraryEnrichment:
    client = OpenLibraryClient()

    return client.match_book(
        title=title,
        author=author,
        isbn=isbn,
    )
