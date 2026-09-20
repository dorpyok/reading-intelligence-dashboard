# Canonical Data Model

## Purpose

The canonical data model defines the source-independent structure used by
Reading Intelligence.

The application may ingest reading data from Goodreads or other future
sources, but downstream analytics, recommendations, reader comparisons,
and AI features should operate on the canonical model rather than directly
on source-specific data.

The goal is to separate:

- Source ingestion
- Canonical reading data
- Enrichment
- Analytics
- Recommendations
- AI-generated explanations

This allows the application to support additional data sources without
redesigning the downstream application.

---

## Design Principles

### 1. Preserve source data

Source-specific data should remain available in the raw ingestion layer.

The canonical model should normalize data needed by the application without
destroying the original source values.

### 2. Separate people, books, and reading relationships

A reader is not a book, and a reader's relationship with a book is not a
property of the book itself.

The model therefore separates:

- Reader
- Book
- Reading Record
- Shelf

### 3. Reading status is derived

Reading status should not be treated as a direct copy of a source shelf.

Different sources may use different labels, and users may create custom
shelves.

The canonical model derives a standardized reading status from available
evidence.

### 4. Preserve user-defined shelves

Source shelves are retained as user data.

Custom shelves may contain valuable signals about a reader's preferences
and should not be discarded simply because they do not map to a standard
reading status.

### 5. External identifiers remain external identifiers

Source IDs such as Goodreads user IDs and Goodreads book IDs should be
stored for traceability.

They should not become the application's internal primary keys.

# Data Flow 

                SOURCE DATA
                    │
                    ▼
             Raw Ingestion
                    │
                    ▼
          Canonical Transformation
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Reader      Book    Reading Record
                              │
                              ▼
                            Shelf
                    │
                    ▼
               Enrichment
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        Genres    Series    Themes
                    │
                    ▼
                Analytics
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Reading DNA  Similarity  Reader Match
                              │
                              ▼
                       Recommendations
                              │
                              ▼
                         AI Explanation
                              │
                              ▼
                       User Application
---

# Entity Model

Reader
  │
  │ 1:M
  ▼
Reading Record
  │
  │ M:1
  ▼
Book

Reading Record
  │
  │ 1:M
  ▼
Shelf

## Tables

reader
├── reader_id          STRING       PRIMARY KEY
├── display_name       STRING
├── source             STRING
├── source_user_id     STRING
└── created_at         DATETIME

book
├── book_id                STRING       PRIMARY KEY
├── title                  STRING
├── author                 STRING
├── isbn                   STRING
├── pages                  INTEGER
├── publication_year       INTEGER
├── description            TEXT
├── cover_url              STRING
├── goodreads_book_id      STRING
└── created_at             DATETIME

reading_record
├── reading_record_id      STRING       PRIMARY KEY
├── reader_id              STRING       FOREIGN KEY → reader.reader_id
├── book_id                STRING       FOREIGN KEY → book.book_id
├── reading_status         STRING
├── rating                 FLOAT
├── date_added             DATE
├── date_read              DATE
├── review                 TEXT
├── source                 STRING
├── source_record_id       STRING
└── ingested_at             DATETIME

shelf
├── shelf_id               STRING       PRIMARY KEY
├── reading_record_id      STRING       FOREIGN KEY → reading_record.reading_record_id
└── shelf_name             STRING

## Reading Status Evidence

Reading status is derived from source evidence rather than assumed from
missing data.

The canonical model recognizes:

- `read`
- `currently_reading`
- `to_read`
- `did_not_finish`
- `unknown`

### Evidence hierarchy

For Goodreads ingestion:

1. Dedicated Goodreads Read shelf membership → `read`
2. `did-not-finish` shelf → `did_not_finish`
3. `currently-reading` shelf → `currently_reading`
4. `to-read` shelf → `to_read`
5. `date_read` may provide evidence of completed reading when present
6. Rating may provide supporting evidence of interaction
7. Otherwise → `unknown`

The exact precedence of overlapping evidence will be implemented in the
transformation layer.

### Important distinctions

`date_read` and `reading_status` are separate concepts.

A book can have:

```text
reading_status = read
date_read = NULL