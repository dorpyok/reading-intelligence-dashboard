import feedparser
import pandas as pd
from pathlib import Path


USERS = {
    "Sarah": "111096646",
    "Shannon": "11074244",
}


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_goodreads_library(user_id):
    records = []

    for page in range(1, 20):
        url = (
            f"https://www.goodreads.com/review/list_rss/"
            f"{user_id}?page={page}&per_page=200"
        )

        print(f"  Pulling page {page}...")

        feed = feedparser.parse(url)

        if not feed.entries:
            break

        print(f"    Books returned: {len(feed.entries)}")

        for entry in feed.entries:
            records.append(
                {
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
                }
            )

    return pd.DataFrame(records)


for user_name, user_id in USERS.items():

    print("\n" + "=" * 60)
    print(f"Pulling {user_name} ({user_id})")
    print("=" * 60)

    df = get_goodreads_library(user_id)

    output_file = OUTPUT_DIR / f"{user_name.lower()}_books.csv"

    df.to_csv(output_file, index=False)

    print(f"\n{user_name} complete!")
    print(f"Records: {len(df)}")
    print(f"Unique Goodreads IDs: {df['source_book_id'].nunique()}")
    print(f"Saved to: {output_file}")