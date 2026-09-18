import os

import feedparser
from dotenv import load_dotenv


# Load settings from the .env file
load_dotenv()

GOODREADS_USER_ID = os.getenv("GOODREADS_USER_ID")


def build_feed_url(
    user_id: str,
    shelf: str = "read",
    page: int = 1,
    per_page: int = 200,
) -> str:
    """Build a Goodreads RSS feed URL."""
    return (
        f"https://www.goodreads.com/review/list_rss/"
        f"{user_id}"
        f"?shelf={shelf}"
        f"&page={page}"
        f"&per_page={per_page}"
    )


def fetch_goodreads_shelf(
    user_id: str,
    shelf: str = "read",
    page: int = 1,
    per_page: int = 200,
):
    """Fetch and parse one page of a Goodreads shelf."""
    url = build_feed_url(
        user_id=user_id,
        shelf=shelf,
        page=page,
        per_page=per_page,
    )

    return feedparser.parse(url)
def fetch_goodreads_library(
    user_id: str,
    page: int = 1,
    per_page: int = 200,
):
    """Fetch and parse one page of a user's Goodreads library RSS feed."""

    url = (
        f"https://www.goodreads.com/review/list_rss/"
        f"{user_id}"
        f"?page={page}"
        f"&per_page={per_page}"
    )

    return feedparser.parse(url)

def main():
    """Test Goodreads library RSS pagination."""

    if not GOODREADS_USER_ID:
        raise ValueError(
            "GOODREADS_USER_ID is not set. "
            "Add it to your .env file."
        )

    print("Goodreads RSS Library Pagination Test")
    print("=" * 45)
    print(f"User ID: {GOODREADS_USER_ID}")
    print("No shelf filter")
    print("Books requested per page: 200")
    print()

    for page in range(1, 4):

        feed = fetch_goodreads_library(
            user_id=GOODREADS_USER_ID,
            page=page,
            per_page=200,
        )

        print(f"PAGE {page}")
        print("-" * 45)
        print(f"Books returned: {len(feed.entries)}")

        if feed.entries:
            print(
                f"First book: {feed.entries[0].get('title')}"
            )

            print(
                f"Last book:  {feed.entries[-1].get('title')}"
            )

        print()

        print("SAMPLE LIBRARY RECORD")
print("=" * 45)

feed = fetch_goodreads_library(
    user_id=GOODREADS_USER_ID,
    page=1,
    per_page=200,
)

if feed.entries:
    book = feed.entries[0]

    print(f"Title: {book.get('title')}")
    print(f"Shelves: {book.get('user_shelves')}")
    print(f"Rating: {book.get('user_rating')}")
    print(f"Read date: {book.get('user_read_at')}")
    
if __name__ == "__main__":
    main()