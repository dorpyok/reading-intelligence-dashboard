import os
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "goodreads_books.csv"

PER_PAGE = 200


def build_feed_url(user_id: str, page: int = 1) -> str:
    return (
        f"https://www.goodreads.com/review/list_rss/"
        f"{user_id}?page={page}&per_page={PER_PAGE}"
    )


def fetch_library_page(user_id: str, page: int):
    url = build_feed_url(user_id, page)
    return feedparser.parse(url)


def parse_book(entry, ingested_at: str) -> dict:
    source_book_id = entry.get("id", "").strip()

    return {
        "source": "goodreads",
        "source_book_id": source_book_id,
        "title": entry.get("title", "").strip(),
        "author": entry.get("author_name", "").strip(),
        "isbn": entry.get("isbn", "").strip(),
        "pages": entry.get("num_pages", ""),
        "user_rating": entry.get("user_rating", ""),
        "average_rating": entry.get("average_rating", ""),
        "date_read": entry.get("user_read_at", ""),
        "date_added": entry.get("user_date_added", ""),
        "date_created": entry.get("user_date_created", ""),
        "shelves": entry.get("user_shelves", ""),
        "description": entry.get("book_description", ""),
        "publication_year": entry.get("published", ""),
        "cover_url": entry.get("book_large_image_url", ""),
        "goodreads_url": entry.get("link", ""),
        "ingested_at": ingested_at,
    }


def fetch_all_books(user_id: str) -> pd.DataFrame:
    all_books = []
    page = 1

    ingested_at = datetime.now(timezone.utc).isoformat()

    while True:
        print(f"Fetching Goodreads library page {page}...")

        feed = fetch_library_page(user_id, page)
        entries = feed.entries

        print(f"  Books returned: {len(entries)}")

        if not entries:
            break

        for entry in entries:
            all_books.append(parse_book(entry, ingested_at))

        page += 1

    return pd.DataFrame(all_books)


def save_books(df: pd.DataFrame) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved {len(df)} books.")
    print(f"Output: {OUTPUT_FILE}")


def main():
    user_id = os.getenv("GOODREADS_USER_ID")

    if not user_id:
        raise ValueError(
            "GOODREADS_USER_ID is not set. "
            "Add it to your .env file."
        )

    print("Goodreads Library Ingestion")
    print("=" * 40)
    print(f"User ID: {user_id}")
    print()

    books = fetch_all_books(user_id)
    save_books(books)


if __name__ == "__main__":
    main()