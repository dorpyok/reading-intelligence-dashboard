from __future__ import annotations

import os
from pathlib import Path

import feedparser
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

USER_ID = os.getenv("GOODREADS_USER_ID")

if not USER_ID:
    raise ValueError("GOODREADS_USER_ID not found in .env")

URL = (
    f"https://www.goodreads.com/review/list_rss/"
    f"{USER_ID}?page=1&per_page=1"
)


def main() -> None:
    print(f"Fetching:\n{URL}\n")

    feed = feedparser.parse(URL)

    print(f"Feed status: {feed.get('status')}")
    print(f"Feed title: {feed.feed.get('title', '')}")
    print(f"Entries returned: {len(feed.entries)}")
    print()

    if not feed.entries:
        print("No entries returned.")
        print("\nFeed bozo:", feed.bozo)
        if feed.bozo_exception:
            print("Feed error:", feed.bozo_exception)
        return

    entry = feed.entries[0]

    print("=" * 80)
    print("ALL RSS ENTRY FIELDS")
    print("=" * 80)

    for key in sorted(entry.keys()):
        value = entry.get(key)

        print(f"\nFIELD: {key}")
        print(f"TYPE:  {type(value).__name__}")
        print(f"VALUE: {value}")

    print("\n" + "=" * 80)
    print("DESCRIPTION-RELATED FIELDS")
    print("=" * 80)

    description_fields = [
        key
        for key in sorted(entry.keys())
        if "description" in key.lower()
        or "summary" in key.lower()
        or "content" in key.lower()
    ]

    if not description_fields:
        print("No description/summary/content fields found.")
    else:
        for key in description_fields:
            print(f"\nFIELD: {key}")
            print("-" * 80)
            print(entry.get(key))

    print("\n" + "=" * 80)
    print("ENTRY AS DICTIONARY")
    print("=" * 80)

    print(dict(entry))


if __name__ == "__main__":
    main()