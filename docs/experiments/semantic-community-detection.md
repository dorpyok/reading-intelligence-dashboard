# Semantic Community Detection Experiment

## Purpose

This experiment establishes the semantic neighborhood representation used by
Reading Intelligence Model C.

The goal is not to force 3,321 books into a small number of broad genres.

The goal is to discover meaningful local neighborhoods in a pooled semantic
book network while allowing weakly connected books to remain isolated.

---

## Frozen semantic representation

Each canonical book is represented using:

- title
- author
- description
- curated Open Library subjects

Embeddings use:

```text
Model: all-MiniLM-L6-v2
Dimensions: 384
Similarity: cosine
```

The semantic embedding matrix is normalized and frozen for the experiment.

Development corpus:

```text
Canonical books: 3,321
Embedding dimensions: 384
Semantic-text coverage: 100%
```

---

## Approaches evaluated

### Flat KMeans

The initial Model C work used KMeans to create a small number of global
interest clusters.

This was useful as a baseline but was rejected as the final semantic
neighborhood representation because:

- it forces every book into one global partition;
- it requires a global K;
- it collapses fine-grained local structure;
- community granularity cannot adapt naturally to different regions of the
  semantic space.

KMeans remains useful as a historical baseline and recommendation experiment,
but it is not the frozen semantic neighborhood model.

### HDBSCAN

HDBSCAN was tested as a density-based alternative.

The resulting structure was not useful for this corpus. The primary
configuration produced a giant central cluster plus a large noise population,
and tuning did not produce a satisfactory global partition.

HDBSCAN was therefore rejected for the current semantic neighborhood layer.

### Global similarity threshold graph

A global threshold without a k-nearest-neighbor constraint was also evaluated.

At useful similarity thresholds, the graph became fragmented enough that it
was not a good representation of the full corpus.

The final approach therefore combines local k-nearest-neighbor candidates with
a similarity threshold.

---

# Final semantic network

The frozen development baseline is:

```text
Graph:
    weighted, symmetrized thresholded kNN

K:
    20

Candidate neighbors:
    up to 20 per book

Similarity threshold:
    cosine >= 0.40

Edge weight:
    cosine similarity

Weak-edge behavior:
    do not force an edge merely to give every book membership

Community algorithm:
    Leiden

Partition:
    RBConfigurationVertexPartition

Resolution:
    3.0
```

The graph is built from the frozen embedding matrix.

The production architecture should eventually replace full pairwise similarity
calculation with an approximate nearest-neighbor index while preserving the
same conceptual contract:

```text
embeddings
    ↓
ANN candidate neighbors
    ↓
retain up to K candidates
    ↓
apply cosine threshold
    ↓
weighted graph
    ↓
Leiden
```

The development full-matrix calculation is a convenience for experimentation,
not the intended production scalability pattern.

---

## Final community structure

The finalized development artifact contains:

```text
Books:                 3,321
Communities:              55
Largest community:       257 books
Median community size:    52 books
Singleton communities:    12
```

There are no forced weak edges merely to eliminate isolated books.

The 12 singleton communities were inspected. Their maximum semantic
similarities were below the 0.40 threshold, so lowering the threshold solely
to eliminate those outliers was rejected.

This preserves the principle that uncertain semantic membership is preferable
to inventing a weak connection.

---

## Manual semantic validation

Representative communities were inspected using the final assignments.

Examples included:

- an extremely coherent Saddle Club community;
- Ice Planet Barbarians;
- Lore Olympus / mythology romance;
- hockey and sports romance;
- missing-person thrillers;
- supernatural/fantasy romance;
- a large Sarah J. Maas / epic fantasy / romantasy neighborhood;
- crime, murder, mystery, and psychological suspense with fantasy/romance
  boundary books.

These inspections support the interpretation that the graph is finding
semantic neighborhoods rather than simply reproducing a small hand-built
genre taxonomy.

Community IDs are arbitrary implementation identifiers. They must not be
treated as permanent semantic labels.

---

## Rejected tuning behavior

The following behaviors are explicitly rejected:

- lowering the similarity threshold solely to eliminate singleton books;
- forcing every book into a community;
- selecting a community count because a dashboard looks cleaner;
- treating community IDs as genres;
- retuning the graph continuously against the same three readers;
- using reader-specific clusters when the product requires comparable
  neighborhoods across readers.

The semantic neighborhood baseline is frozen before continuing Reading DNA
development.

---

## Reproducibility

The finalized artifacts are stored under:

```text
data/processed/canonical/leiden/
├── leiden_assignments.csv
├── leiden_community_summary.csv
└── ...
```

The assignment artifact contains one row per canonical book with its current
community assignment and semantic metadata.

The community summary contains community size and internal similarity
statistics.

The final inspector reads these finalized artifacts directly and is
read-only; it does not regenerate the model.

---

## Production principle

The semantic graph is a discovery layer.

It is not the final user-facing taxonomy.

Reading DNA should describe a reader using evidence across semantic
neighborhoods and overlapping attributes rather than telling the reader:

> "You are genre X."

The product goal is:

> "This actually describes how I read."

---

## Status

**Frozen development baseline**

The semantic representation and community detection configuration should not be
retuned during the current Reading DNA scoring work.

Future changes should be treated as a new experiment with a new artifact and
explicit comparison against this baseline.
