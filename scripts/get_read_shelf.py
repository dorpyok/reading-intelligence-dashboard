import feedparser
import pandas as pd
from pathlib import Path


GOODREADS_USER_ID = "63553279"
READ_SHELF_URL = (
    f"https://www.goodreads.com/review/list_rss/"
    f"{GOODREADS_USER_ID}?shelf=read&per_page=200"
)

OUTPUT_FILE = Path("data/raw/goodreads_read_books.csv")


def get_read_shelf():
    print("Pulling Goodreads Read shelf...")
    print(READ_SHELF_URL)

    feed = feedparser.parse(READ_SHELF_URL)

    print(f"Books returned: {len(feed.entries)}")

    records = []

    for entry in feed.entries:
        records.append(
            {
                "goodreads_book_id": entry.get("book_id"),
                "title": entry.get("title"),
                "author": entry.get("author_name"),
                "isbn": entry.get("isbn"),
                "pages": entry.get("num_pages"),
                "user_rating": entry.get("user_rating"),
                "date_read": entry.get("user_read_at"),
                "date_added": entry.get("user_date_added"),
                "review": entry.get("user_review"),
                "shelves": entry.get("user_shelves"),
            }
        )

    df = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Records saved: {len(df)}")


if __name__ == "__main__":
    get_read_shelf()