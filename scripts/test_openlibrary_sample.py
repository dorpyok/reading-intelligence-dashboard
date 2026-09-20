"""
Open Library 40-book diagnostic.

Purpose:
- Start with real Goodreads records from data/raw/goodreads_books.csv
- Match Goodreads books to Open Library Works
- Preserve the existing matching hierarchy
- Add an author-first title scan as a fallback
- Enrich matched Works with subjects
- Inspect all Work editions for additional subjects
- Record matching provenance and diagnostics

Matching hierarchy:
    1. Exact ISBN
    2. Title + author search with fuzzy scoring
    3. Author-first search -> scan author's works -> fuzzy title match
    4. Strict title-only fallback
    5. Unmatched

IMPORTANT:
- No book titles or Open Library IDs are hard-coded.
- TEST_LIMIT simply selects the first N real Goodreads records.
"""

import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from urllib.parse import quote

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOODREADS_FILE = PROJECT_ROOT / "data" / "raw" / "goodreads_books.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "openlibrary_diagnostic"
OUTPUT_FILE = OUTPUT_DIR / "openlibrary_40_book_results.json"

CACHE_DIR = OUTPUT_DIR / "cache"

OPENLIBRARY_BASE = "https://openlibrary.org"

TEST_LIMIT = 40

REQUEST_DELAY = 0.15

TITLE_AUTHOR_THRESHOLD = 0.70
AUTHOR_SCAN_TITLE_THRESHOLD = 0.82
TITLE_ONLY_THRESHOLD = 0.92

AUTHOR_PAGE_SIZE = 100
MAX_AUTHOR_WORKS = 500


# ============================================================
# HTTP / CACHE
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "ReadingIntelligenceDashboard/0.1 "
            "(open-source reading analytics project)"
        )
    }
)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def cached_get_json(url: str, cache_key: str):
    """
    GET JSON with a simple local cache.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE_DIR / f"{safe_filename(cache_key)}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    time.sleep(REQUEST_DELAY)

    response = session.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


# ============================================================
# NORMALIZATION
# ============================================================

def clean_text(value) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_text(value: str) -> str:
    """
    General text normalization.
    """

    value = clean_text(value).lower()

    value = value.replace("&", " and ")

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def remove_parenthetical(value: str) -> str:
    """
    Removes trailing parenthetical text.

    Example:
        The September House (A Novel)
        ->
        The September House
    """

    value = clean_text(value)

    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)

    return value.strip()


def remove_subtitle(value: str) -> str:
    """
    Removes subtitle after colon.

    Example:
        Salt: A World History
        ->
        Salt
    """

    value = clean_text(value)

    if ":" in value:
        value = value.split(":", 1)[0]

    return value.strip()


def normalize_article_position(value: str) -> str:
    """
    Normalizes leading/trailing English articles.

    Examples:

        The One That Got Away
            ->
        one that got away

        One That Got Away, The
            ->
        one that got away

        A Woman of No Importance
            ->
        woman of no importance

        Woman of No Importance, A
            ->
        woman of no importance
    """

    value = normalize_text(value)

    articles = {"the", "a", "an"}

    words = value.split()

    if not words:
        return ""

    # Leading article
    if words[0] in articles:
        words = words[1:]

    # Trailing article
    if words and words[-1] in articles:
        words = words[:-1]

    return " ".join(words)


def title_variants(title: str) -> list[str]:
    """
    Generate several normalized title representations.
    """

    original = clean_text(title)

    variants = set()

    candidates = [
        original,
        remove_parenthetical(original),
    ]

    for candidate in list(candidates):
        candidates.append(remove_subtitle(candidate))

    for candidate in candidates:
        normalized = normalize_text(candidate)

        if normalized:
            variants.add(normalized)

        article_normalized = normalize_article_position(candidate)

        if article_normalized:
            variants.add(article_normalized)

    return sorted(variants)


def title_similarity(title_a: str, title_b: str) -> float:
    """
    Compare titles using multiple normalized representations.

    The best pairwise similarity is returned.

    This allows:

        The One That Got Away

    to match:

        One That Got Away, The
    """

    variants_a = title_variants(title_a)
    variants_b = title_variants(title_b)

    if not variants_a or not variants_b:
        return 0.0

    best = 0.0

    for a in variants_a:
        for b in variants_b:
            score = SequenceMatcher(None, a, b).ratio()

            if score > best:
                best = score

    return best


def author_similarity(author_a: str, author_b: str) -> float:
    """
    Compare author names.
    """

    a = normalize_text(author_a)
    b = normalize_text(author_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ============================================================
# SEARCH
# ============================================================

def search_by_title_author(title: str, author: str, limit: int = 10):
    """
    Search Open Library using title + author.

    Returns Work-level search results.
    """

    params = (
        f"title={quote(title)}"
        f"&author={quote(author)}"
        f"&limit={limit}"
        f"&fields=key,title,author_name,author_key,edition_count"
    )

    url = f"{OPENLIBRARY_BASE}/search.json?{params}"

    cache_key = (
        f"title_author_"
        f"{normalize_text(title)}_"
        f"{normalize_text(author)}"
    )

    return cached_get_json(url, cache_key)


def search_by_author(author: str, limit: int = 100, offset: int = 0):
    """
    Search Open Library by author.

    This is the NEW fallback.

    It deliberately retrieves Work-level fields only.
    """

    params = (
        f"author={quote(author)}"
        f"&limit={limit}"
        f"&offset={offset}"
        f"&fields=key,title,author_name,author_key,edition_count"
    )

    url = f"{OPENLIBRARY_BASE}/search.json?{params}"

    cache_key = (
        f"author_search_"
        f"{normalize_text(author)}_"
        f"{offset}"
    )

    return cached_get_json(url, cache_key)


def search_by_title(title: str, limit: int = 10):
    """
    Strict title-only fallback.
    """

    params = (
        f"title={quote(title)}"
        f"&limit={limit}"
        f"&fields=key,title,author_name,author_key,edition_count"
    )

    url = f"{OPENLIBRARY_BASE}/search.json?{params}"

    cache_key = f"title_only_{normalize_text(title)}"

    return cached_get_json(url, cache_key)


# ============================================================
# ISBN MATCH
# ============================================================

def lookup_isbn(isbn: str):
    """
    Exact ISBN lookup.

    Returns Edition JSON when available.
    """

    isbn = clean_text(isbn)

    if not isbn:
        return None

    isbn = re.sub(r"[^0-9Xx]", "", isbn)

    if not isbn:
        return None

    url = f"{OPENLIBRARY_BASE}/isbn/{isbn}.json"

    cache_key = f"isbn_{isbn}"

    try:
        return cached_get_json(url, cache_key)
    except requests.HTTPError:
        return None


def extract_work_from_edition(edition_data):
    """
    Extract Work ID from an Open Library Edition response.
    """

    if not edition_data:
        return None

    works = edition_data.get("works", [])

    if not works:
        return None

    first_work = works[0]

    key = first_work.get("key")

    if not key:
        return None

    return key.rsplit("/", 1)[-1]


# ============================================================
# WORK DATA
# ============================================================

def fetch_work(work_id: str):
    """
    Fetch Work JSON.
    """

    url = f"{OPENLIBRARY_BASE}/works/{work_id}.json"

    cache_key = f"work_{work_id}"

    return cached_get_json(url, cache_key)


def fetch_work_editions(work_id: str):
    """
    Retrieve all available editions for a Work.

    Uses Open Library's documented pagination structure.
    """

    editions = []

    offset = 0
    limit = 100

    while True:

        url = (
            f"{OPENLIBRARY_BASE}/works/"
            f"{work_id}/editions.json"
            f"?limit={limit}&offset={offset}"
        )

        cache_key = f"editions_{work_id}_{offset}"

        data = cached_get_json(url, cache_key)

        entries = data.get("entries", [])

        if not entries:
            break

        editions.extend(entries)

        if len(entries) < limit:
            break

        offset += limit

    return editions


# ============================================================
# SUBJECT EXTRACTION
# ============================================================

def extract_subjects_from_work(work_data):
    """
    Extract Work-level subjects.
    """

    if not work_data:
        return []

    subjects = []

    raw_subjects = work_data.get("subjects", [])

    if isinstance(raw_subjects, list):
        subjects.extend(raw_subjects)

    return subjects


def extract_subjects_from_edition(edition):
    """
    Extract edition-level subjects.

    Open Library can expose several subject-like fields.
    """

    if not edition:
        return []

    subjects = []

    for field in [
        "subjects",
        "subject",
        "subject_facet",
    ]:
        value = edition.get(field)

        if isinstance(value, list):
            subjects.extend(value)

    return subjects


def normalize_subject(subject):
    """
    Convert subject values to clean strings.
    """

    if isinstance(subject, str):
        return subject.strip()

    if isinstance(subject, dict):
        name = subject.get("name")

        if name:
            return str(name).strip()

    return ""


def deduplicate_subjects(subjects):
    """
    Case-insensitive subject deduplication while preserving
    the first encountered representation.
    """

    seen = set()
    result = []

    for subject in subjects:

        subject = normalize_subject(subject)

        if not subject:
            continue

        key = subject.lower()

        if key not in seen:
            seen.add(key)
            result.append(subject)

    return result


# ============================================================
# MATCHING
# ============================================================

def choose_title_author_match(
    goodreads_title: str,
    goodreads_author: str,
    docs: list,
):
    """
    Select the strongest title+author search result.

    Title score = 60%
    Author score = 40%
    """

    candidates = []

    for doc in docs:

        ol_title = clean_text(doc.get("title"))

        author_names = doc.get("author_name") or []

        ol_author = author_names[0] if author_names else ""

        if not ol_title:
            continue

        title_score = title_similarity(
            goodreads_title,
            ol_title,
        )

        author_score = author_similarity(
            goodreads_author,
            ol_author,
        )

        combined_score = (
            0.60 * title_score
            + 0.40 * author_score
        )

        candidates.append(
            {
                "doc": doc,
                "title_score": round(title_score, 4),
                "author_score": round(author_score, 4),
                "combined_score": round(combined_score, 4),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    best = candidates[0]

    if best["combined_score"] < TITLE_AUTHOR_THRESHOLD:
        return None

    return best


def author_first_match(
    goodreads_title: str,
    goodreads_author: str,
):
    """
    NEW FALLBACK.

    Search Open Library for the Goodreads author, then scan that
    author's Work titles for a strong title match.

    This handles title formatting differences such as:

        The One That Got Away

    vs.

        One That Got Away, The

    No Work ID or title is hard-coded.
    """

    if not goodreads_author:
        return None

    all_docs = []

    offset = 0

    while offset < MAX_AUTHOR_WORKS:

        data = search_by_author(
            goodreads_author,
            limit=AUTHOR_PAGE_SIZE,
            offset=offset,
        )

        docs = data.get("docs", [])

        if not docs:
            break

        all_docs.extend(docs)

        num_found = data.get("num_found")

        if num_found is not None:
            if offset + len(docs) >= num_found:
                break

        if len(docs) < AUTHOR_PAGE_SIZE:
            break

        offset += AUTHOR_PAGE_SIZE

    candidates = []

    for doc in all_docs:

        work_id = doc.get("key")

        if not work_id:
            continue

        work_id = work_id.rsplit("/", 1)[-1]

        ol_title = clean_text(doc.get("title"))

        if not ol_title:
            continue

        author_names = doc.get("author_name") or []

        if not author_names:
            continue

        # Require an actual author relationship.
        best_author_score = max(
            (
                author_similarity(
                    goodreads_author,
                    author_name,
                )
                for author_name in author_names
                if clean_text(author_name)
            ),
            default=0.0,
        )

        if best_author_score < 0.85:
            continue

        title_score = title_similarity(
            goodreads_title,
            ol_title,
        )

        if title_score < AUTHOR_SCAN_TITLE_THRESHOLD:
            continue

        candidates.append(
            {
                "work_id": work_id,
                "title": ol_title,
                "author_names": author_names,
                "title_score": round(title_score, 4),
                "author_score": round(best_author_score, 4),
                "edition_count": doc.get("edition_count"),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["title_score"],
            x["author_score"],
        ),
        reverse=True,
    )

    return candidates[0]


def strict_title_only_match(
    goodreads_title: str,
):
    """
    Final fallback.

    This is intentionally strict because title-only matches can
    easily be ambiguous.
    """

    data = search_by_title(
        goodreads_title,
        limit=10,
    )

    docs = data.get("docs", [])

    candidates = []

    for doc in docs:

        ol_title = clean_text(doc.get("title"))

        if not ol_title:
            continue

        score = title_similarity(
            goodreads_title,
            ol_title,
        )

        if score >= TITLE_ONLY_THRESHOLD:

            candidates.append(
                {
                    "doc": doc,
                    "title_score": round(score, 4),
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["title_score"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# SINGLE BOOK MATCH
# ============================================================

def match_book(row):
    """
    Match one Goodreads record to an Open Library Work.
    """

    title = clean_text(row.get("title"))
    author = clean_text(row.get("author"))
    isbn = clean_text(row.get("isbn"))

    result = {
        "goodreads_title": title,
        "goodreads_author": author,
        "goodreads_isbn": isbn,
        "match_method": None,
        "match_score": None,
        "title_score": None,
        "author_score": None,
        "work_id": None,
        "openlibrary_title": None,
        "openlibrary_authors": [],
        "edition_count": None,
    }

    # --------------------------------------------------------
    # 1. Exact ISBN
    # --------------------------------------------------------

    if isbn:

        edition = lookup_isbn(isbn)

        if edition:

            work_id = extract_work_from_edition(
                edition
            )

            if work_id:

                result.update(
                    {
                        "match_method": "isbn",
                        "work_id": work_id,
                        "openlibrary_title": edition.get(
                            "title"
                        ),
                        "openlibrary_authors": [
                            author.get("name", "")
                            for author in edition.get(
                                "authors",
                                []
                            )
                            if isinstance(author, dict)
                        ],
                    }
                )

                return result

    # --------------------------------------------------------
    # 2. Title + Author
    # --------------------------------------------------------

    if title and author:

        data = search_by_title_author(
            title,
            author,
            limit=10,
        )

        docs = data.get("docs", [])

        best = choose_title_author_match(
            title,
            author,
            docs,
        )

        if best:

            doc = best["doc"]

            work_id = doc.get("key")

            if work_id:

                work_id = work_id.rsplit(
                    "/",
                    1,
                )[-1]

                result.update(
                    {
                        "match_method": "author_fuzzy_title",
                        "match_score": best[
                            "combined_score"
                        ],
                        "title_score": best[
                            "title_score"
                        ],
                        "author_score": best[
                            "author_score"
                        ],
                        "work_id": work_id,
                        "openlibrary_title": doc.get(
                            "title"
                        ),
                        "openlibrary_authors": doc.get(
                            "author_name",
                            [],
                        ),
                        "edition_count": doc.get(
                            "edition_count"
                        ),
                    }
                )

                return result

    # --------------------------------------------------------
    # 3. NEW: Author-first title scan
    # --------------------------------------------------------

    if title and author:

        best = author_first_match(
            title,
            author,
        )

        if best:

            result.update(
                {
                    "match_method": "author_scan_title",
                    "match_score": best[
                        "title_score"
                    ],
                    "title_score": best[
                        "title_score"
                    ],
                    "author_score": best[
                        "author_score"
                    ],
                    "work_id": best[
                        "work_id"
                    ],
                    "openlibrary_title": best[
                        "title"
                    ],
                    "openlibrary_authors": best[
                        "author_names"
                    ],
                    "edition_count": best.get(
                        "edition_count"
                    ),
                }
            )

            return result

    # --------------------------------------------------------
    # 4. Strict title-only
    # --------------------------------------------------------

    if title:

        best = strict_title_only_match(
            title,
        )

        if best:

            doc = best["doc"]

            work_id = doc.get("key")

            if work_id:

                work_id = work_id.rsplit(
                    "/",
                    1,
                )[-1]

                result.update(
                    {
                        "match_method": "title_only",
                        "match_score": best[
                            "title_score"
                        ],
                        "title_score": best[
                            "title_score"
                        ],
                        "work_id": work_id,
                        "openlibrary_title": doc.get(
                            "title"
                        ),
                        "openlibrary_authors": doc.get(
                            "author_name",
                            [],
                        ),
                        "edition_count": doc.get(
                            "edition_count"
                        ),
                    }
                )

                return result

    # --------------------------------------------------------
    # 5. Unmatched
    # --------------------------------------------------------

    return result


# ============================================================
# ENRICH ONE MATCH
# ============================================================

def enrich_match(match):
    """
    Fetch Work-level and Edition-level metadata for a matched
    Work.
    """

    work_id = match.get("work_id")

    enrichment = {
        "work_subjects": [],
        "edition_subjects": [],
        "all_subjects": [],
        "subject_count": 0,
        "edition_count_actual": 0,
        "work_fetch_error": None,
        "edition_fetch_error": None,
    }

    if not work_id:
        return enrichment

    # --------------------------------------------------------
    # Work
    # --------------------------------------------------------

    try:

        work_data = fetch_work(work_id)

        work_subjects = (
            extract_subjects_from_work(
                work_data
            )
        )

        enrichment["work_subjects"] = [
            normalize_subject(subject)
            for subject in work_subjects
            if normalize_subject(subject)
        ]

    except Exception as exc:

        enrichment["work_fetch_error"] = str(exc)

    # --------------------------------------------------------
    # Editions
    # --------------------------------------------------------

    try:

        editions = fetch_work_editions(
            work_id
        )

        enrichment[
            "edition_count_actual"
        ] = len(editions)

        edition_subjects = []

        for edition in editions:

            edition_subjects.extend(
                extract_subjects_from_edition(
                    edition
                )
            )

        enrichment["edition_subjects"] = [
            normalize_subject(subject)
            for subject in edition_subjects
            if normalize_subject(subject)
        ]

    except Exception as exc:

        enrichment[
            "edition_fetch_error"
        ] = str(exc)

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    enrichment["all_subjects"] = (
        deduplicate_subjects(
            enrichment["work_subjects"]
            + enrichment["edition_subjects"]
        )
    )

    enrichment["subject_count"] = len(
        enrichment["all_subjects"]
    )

    return enrichment


# ============================================================
# DIAGNOSTICS
# ============================================================

def calculate_summary(results):
    total = len(results)

    isbn_records = sum(
        bool(r["goodreads_isbn"])
        for r in results
    )

    isbn_matches = sum(
        r["match_method"] == "isbn"
        for r in results
    )

    author_fuzzy_matches = sum(
        r["match_method"]
        == "author_fuzzy_title"
        for r in results
    )

    author_scan_matches = sum(
        r["match_method"]
        == "author_scan_title"
        for r in results
    )

    title_only_matches = sum(
        r["match_method"]
        == "title_only"
        for r in results
    )

    unmatched = sum(
        r["match_method"] is None
        for r in results
    )

    work_matches = sum(
        bool(r["work_id"])
        for r in results
    )

    with_work_subjects = sum(
        len(r["work_subjects"]) > 0
        for r in results
    )

    edition_only_subjects = sum(
        len(r["work_subjects"]) == 0
        and len(r["edition_subjects"]) > 0
        for r in results
    )

    matched_no_subjects = sum(
        bool(r["work_id"])
        and len(r["all_subjects"]) == 0
        for r in results
    )

    raw_work_subjects = sum(
        len(r["work_subjects"])
        for r in results
    )

    raw_edition_subjects = sum(
        len(r["edition_subjects"])
        for r in results
    )

    deduplicated_subjects = sum(
        len(r["all_subjects"])
        for r in results
    )

    return {
        "books_tested": total,

        "records_with_isbn": isbn_records,

        "isbn_matches": isbn_matches,

        "isbn_match_rate": (
            round(
                isbn_matches
                / isbn_records
                * 100,
                1,
            )
            if isbn_records
            else 0
        ),

        "author_fuzzy_matches": (
            author_fuzzy_matches
        ),

        "author_scan_title_matches": (
            author_scan_matches
        ),

        "title_only_matches": (
            title_only_matches
        ),

        "unmatched": unmatched,

        "work_matches": work_matches,

        "work_match_rate": (
            round(
                work_matches
                / total
                * 100,
                1,
            )
            if total
            else 0
        ),

        "with_work_subjects": (
            with_work_subjects
        ),

        "edition_only_subjects": (
            edition_only_subjects
        ),

        "matched_no_subjects": (
            matched_no_subjects
        ),

        "subject_coverage": (
            round(
                (
                    with_work_subjects
                    + edition_only_subjects
                )
                / work_matches
                * 100,
                1,
            )
            if work_matches
            else 0
        ),

        "raw_work_subjects": (
            raw_work_subjects
        ),

        "raw_edition_subjects": (
            raw_edition_subjects
        ),

        "deduplicated_subjects": (
            deduplicated_subjects
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OPEN LIBRARY MATCHING DIAGNOSTIC")
    print("=" * 70)

    print()
    print(f"Goodreads source: {GOODREADS_FILE}")
    print()

    # --------------------------------------------------------
    # Load actual Goodreads data
    # --------------------------------------------------------

    goodreads_df = pd.read_csv(
        GOODREADS_FILE,
        dtype=str,
    )

    print(
        f"Goodreads records available: "
        f"{len(goodreads_df)}"
    )

    if TEST_LIMIT is None:
        test_df = goodreads_df.copy()
    else:
        test_df = goodreads_df.head(
            TEST_LIMIT
        ).copy()

    print(
        f"Records being tested: "
        f"{len(test_df)}"
    )

    print()

    # --------------------------------------------------------
    # Match + enrich
    # --------------------------------------------------------

    results = []

    for index, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):

        title = clean_text(
            row.get("title")
        )

        author = clean_text(
            row.get("author")
        )

        print(
            f"[{index}/{len(test_df)}] "
            f"{title} — {author}"
        )

        try:

            match = match_book(row)

            enrichment = enrich_match(
                match
            )

            result = {
                **match,
                **enrichment,
            }

            results.append(result)

            print(
                f"    Match: "
                f"{result['match_method']}"
            )

            print(
                f"    Work: "
                f"{result['work_id']}"
            )

            print(
                f"    Subjects: "
                f"{result['subject_count']}"
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            results.append(
                {
                    "goodreads_title": title,
                    "goodreads_author": author,
                    "goodreads_isbn": clean_text(
                        row.get("isbn")
                    ),
                    "match_method": None,
                    "match_score": None,
                    "title_score": None,
                    "author_score": None,
                    "work_id": None,
                    "openlibrary_title": None,
                    "openlibrary_authors": [],
                    "edition_count": None,
                    "work_subjects": [],
                    "edition_subjects": [],
                    "all_subjects": [],
                    "subject_count": 0,
                    "edition_count_actual": 0,
                    "work_fetch_error": None,
                    "edition_fetch_error": None,
                    "error": str(exc),
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = calculate_summary(
        results
    )

    output = {
        "configuration": {
            "test_limit": TEST_LIMIT,
            "title_author_threshold": (
                TITLE_AUTHOR_THRESHOLD
            ),
            "author_scan_title_threshold": (
                AUTHOR_SCAN_TITLE_THRESHOLD
            ),
            "title_only_threshold": (
                TITLE_ONLY_THRESHOLD
            ),
            "author_page_size": (
                AUTHOR_PAGE_SIZE
            ),
            "max_author_works": (
                MAX_AUTHOR_WORKS
            ),
        },
        "summary": summary,
        "results": results,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print()
    print(
        f"Saved results to:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Print newly recovered author-scan matches
    # --------------------------------------------------------

    author_scan_results = [
        r
        for r in results
        if r["match_method"]
        == "author_scan_title"
    ]

    if author_scan_results:

        print()
        print(
            "=" * 70
        )
        print(
            "AUTHOR-SCAN MATCHES"
        )
        print(
            "=" * 70
        )

        for result in author_scan_results:

            print(
                f"{result['goodreads_title']}"
            )

            print(
                f"  Goodreads author: "
                f"{result['goodreads_author']}"
            )

            print(
                f"  Open Library title: "
                f"{result['openlibrary_title']}"
            )

            print(
                f"  Work ID: "
                f"{result['work_id']}"
            )

            print(
                f"  Title score: "
                f"{result['title_score']}"
            )

            print(
                f"  Author score: "
                f"{result['author_score']}"
            )

            print()


if __name__ == "__main__":
    main()