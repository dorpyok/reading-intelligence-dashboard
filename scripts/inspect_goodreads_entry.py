import feedparser
from pprint import pprint


GOODREADS_USER_ID = "63553279"
TARGET_TITLE = "A Wrinkle in Time (Time Quintet, #1)"

for page in range(1, 10):

    url = (
        f"https://www.goodreads.com/review/list_rss/"
        f"{GOODREADS_USER_ID}?page={page}&per_page=200"
    )

    print(f"Checking page {page}...")

    feed = feedparser.parse(url)

    print(f"  Books returned: {len(feed.entries)}")

    if not feed.entries:
        break

    for entry in feed.entries:
        if entry.get("title") == TARGET_TITLE:
            print("\nFOUND BOOK!\n")
            pprint(dict(entry))
            raise SystemExit

print(f"\nCould not find: {TARGET_TITLE}")