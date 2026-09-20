# Goodreads Ingestion Findings

## Purpose

This document records findings from validating the Goodreads RSS ingestion
pipeline against multiple Goodreads users.

The goal is to identify source-system behavior before designing the
source-agnostic canonical reading data model.

## Validation Users

The ingestion pipeline was tested against three Goodreads accounts.

| Reader | RSS Records | Unique Book IDs | Duplicate IDs |
|---|---:|---:|---:|
| Reader A | 359 | 359 | 0 |
| Reader B | 1,866 | 1,866 | 0 |
| Reader C | 1,330 | 1,330 | 0 |

## Findings

### 1. RSS pagination works across different library sizes

The ingestion pipeline successfully retrieved libraries ranging from
359 records to 1,866 records.

The pipeline does not depend on a fixed number of pages. It continues
requesting pages until Goodreads returns an empty page.

### 2. Goodreads book IDs were unique within each tested library

No duplicate Goodreads `book_id` values were found in any of the three
tested libraries.

This means the raw ingestion layer should not perform arbitrary
deduplication.

Book identity can instead be handled by the canonical data model.

### 3. Shelf values are user-specific

Shelf structures differed significantly between users.

Examples included:

- `to-read`
- `currently-reading`
- `did-not-finish`
- `saddleclub`
- `you-literally-own-this-book-read-it`

Some records contained multiple shelf values, for example:

- `to-read, you-literally-own-this-book-read-it`
- `you-literally-own-this-book-read-it, to-read`

Some records had no shelf value.

### 4. Shelves should not be treated as a single status field

The `user_shelves` field represents a collection of user-defined labels,
not a standardized reading status.

The canonical model should preserve source shelves separately from any
derived reading status.

### 5. Raw ingestion should preserve source values

The ingestion layer should capture Goodreads values without applying
business interpretation.

For example, Goodreads uses a `user_rating` value of `0` for books
that have not been rated.

The raw layer should preserve this value.

The canonical layer can later interpret `0` as a missing/unrated value.

### 6. Reading status should be derived independently

A book's Goodreads shelf should not be the sole source of truth for
whether the book has been completed.

Fields such as `user_read_at`, current-reading information, and source
shelves should be evaluated when creating a canonical reading status.

## Architectural Implications

The ingestion architecture should remain source-specific:

    Goodreads RSS
          |
          v
    Raw Goodreads Records
          |
          v
    Canonical Reading Model
          |
          +------------------+
          |                  |
          v                  v
    Reader Analytics     Reader Match

The canonical model should separate:

- Reader
- Book
- Reading Record
- User Shelf

A Book represents the underlying work.

A Reading Record represents one reader's relationship with that book.

This separation is necessary to support comparing multiple readers.

## Current Decision

The Goodreads ingestion layer is considered validated against multiple
users and multiple library sizes.

No additional source-specific transformation will be added to the raw
ingestion layer at this stage.

## Updated: Reading Status and Goodreads Read Shelf

### Finding

The Goodreads general library RSS feed does not reliably expose a user's
exclusive `read` shelf through the `user_shelves` field.

For example, a book confirmed to be on the user's Goodreads Read shelf may
appear in the general library RSS with:

- `user_shelves` = blank
- `user_read_at` = blank
- `user_rating` = populated or blank
- `user_date_added` = populated

Therefore:

- blank `user_shelves` must NOT be interpreted as `read`
- `user_date_added` must NOT be interpreted as the date the book was read
- `user_rating` must NOT be the primary definition of `read`

### Validation

The dedicated Goodreads Read-shelf RSS feed was compared against the books
classified as `unknown` from the general library RSS.

For the validated reader:

- General library records: 359
- Dedicated Read-shelf records: 139
- Previously `unknown` records: 48
- Previously `unknown` records found on the Read shelf: 48

This confirmed that books can be marked as read without having a reading date,
rating, review, or visible `read` value in the general library RSS shelf field.

### Ingestion Decision

The application will use one Goodreads ingestion workflow per user.

The workflow will:

1. Pull the user's general Goodreads library RSS feed.
2. Preserve the complete raw library records.
3. Pull the user's Goodreads Read shelf as a lightweight status lookup.
4. Match Read-shelf Goodreads book IDs to library records.
5. Use Read-shelf membership as authoritative evidence for `read` status.
6. Continue preserving all other Goodreads shelf values as user-defined shelf
   information.

The Read-shelf lookup is part of the same ingestion workflow; it is not a
second user-facing ingestion pipeline.

### Reading Date

`date_read` represents a reading date only when Goodreads provides one.

`date_added` represents when the book was added to the Goodreads library and
must not be interpreted as the date the book was read.

A user may add a book to Goodreads after having read it previously, without
recording the original reading date.

### Ratings and Reviews

Ratings and reviews are preserved as separate evidence.

A rating is useful preference information but is not the authoritative
definition of reading status.

A completed book may legitimately have:

- no rating
- no review
- no reading date

Such a book remains valuable analytical data if its read status is established
through the Read shelf.

### Custom Shelves

Goodreads users may create custom shelves and may place books on multiple
shelves.

Custom shelves must be preserved exactly as source data.

The application will not automatically interpret a custom shelf containing
words such as `read` as a completed-reading status. Custom shelf semantics are
user-defined and should not be assumed to have universal meaning.

### Uncertain Reading Records

Books that cannot be confidently identified as read from the available source
evidence will not be discarded.

They will remain in the canonical dataset and may later be surfaced by the
application as `Needs confirmation`.

This allows the application to preserve potentially valuable reading history
without silently making unsupported assumptions.

### Analytical Principle

The ingestion layer preserves source evidence.

The analytical layer determines how that evidence should be used.

A book with no rating or review is not necessarily unimportant. It may still
provide useful evidence of:

- books the reader has encountered
- authors the reader has read
- series exposure
- reading history
- publication and book metadata

Preference strength and reading status are therefore treated as separate
analytical concepts.