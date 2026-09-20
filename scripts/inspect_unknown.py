import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "data" / "reading_intelligence.db"


connection = sqlite3.connect(DATABASE_FILE)

query = """
SELECT
    b.title,
    b.author,
    rr.date_read,
    rr.date_added,
    rr.rating,
    rr.review
FROM reading_record rr
JOIN book b
    ON rr.book_id = b.book_id
WHERE rr.reading_status = 'unknown'
ORDER BY b.title;
"""

rows = connection.execute(query).fetchall()

print(f"Unknown books: {len(rows)}")
print()

for row in rows:
    title, author, date_read, date_added, rating, review = row

    print(f"Title:       {title}")
    print(f"Author:      {author}")
    print(f"Date read:   {date_read}")
    print(f"Date added:  {date_added}")
    print(f"Rating:      {rating}")
    print(f"Review:      {review}")
    print("-" * 80)

connection.close()