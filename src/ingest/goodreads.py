import os
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests
from dotenv import load_dotenv


# Load settings from the .env file
load_dotenv()

GOODREADS_USER_ID = os.getenv("GOODREADS_USER_ID")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "goodreads_books.csv"

PER_PAGE = 200


def build_feed_url(
    user_id: str,
    page: int = 1,
    per_page: int = PER_PAGE,
) -> str:
    """Build the Goodreads RSS library URL."""

    return (
        f"https://www.goodreads.com/review/list_rss/"
        f"{user_id}"
        f"?page={page}"
        f"&per_page={per_page}"
    )


def fetch_library_page(
    user_id: str,
    page: int,
    per_page: int = PER_PAGE,
):
    """Fetch and parse one page of a Goodreads library."""

    url = build_feed_url(
        user_id=user_id,
        page=page,
        per_page=per_page,
    )

    return feedparser.parse(url)

def parse_book(entry: dict, ingested_at: str) -> dict:
    """Convert one Goodreads RSS entry into a structured record."""

    return {
        "source": "goodreads",
        "source_book_id": entry.get("book_id"),
        "title": entry.get("title"),
        "author": entry.get("author_name"),
        "isbn": entry.get("isbn"),
        "pages": entry.get("num_pages"),
        "user_rating": entry.get("user_rating"),
        "average_rating": entry.get("average_rating"),
        "date_read": entry.get("user_read_at"),
        "date_added": entry.get("user_date_added"),
        "date_created": entry.get("user_date_created"),
        "shelves": entry.get("user_shelves"),
        "description": entry.get("book_description"),
        "publication_year": entry.get("book_published"),
        "cover_url": entry.get("book_large_image_url"),
        "goodreads_url": entry.get("link"),
        "ingested_at": ingested_at,
    }


def fetch_all_books(user_id: str) -> list[dict]:
    """Fetch every book in a Goodreads library."""

    all_books = []
    page = 1

    ingested_at = datetime.now(timezone.utc).isoformat()

    while True:
        print(f"Fetching Goodreads page {page}...")

        feed = fetch_library_page(
            user_id=user_id,
            page=page,
            per_page=PER_PAGE,
        )

        books_on_page = feed.entries

        print(f"  Books returned: {len(books_on_page)}")

        if not books_on_page:
            break

        for entry in books_on_page:
            book = parse_book(
                entry=entry,
                ingested_at=ingested_at,
            )

            all_books.append(book)

        page += 1

    return all_books


def save_books(books: list[dict]) -> None:
    """Save Goodreads books to the raw data directory."""

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(books)

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved {len(dataframe)} books.")
    print(f"Output: {OUTPUT_FILE}")


def main():
    """Run the Goodreads ingestion pipeline."""

    if not GOODREADS_USER_ID:
        raise ValueError(
            "GOODREADS_USER_ID is not set. "
            "Add it to your .env file."
        )

    print("Goodreads Library Ingestion")
    print("=" * 40)
    print(f"User ID: {GOODREADS_USER_ID}")
    print()

    books = fetch_all_books(
        user_id=GOODREADS_USER_ID,
    )

    save_books(books)


if __name__ == "__main__":
    main()