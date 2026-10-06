# Model C — Semantic Reading DNA

## Purpose

Model C is the semantic representation layer for the Reading Intelligence Dashboard.

Its job is to answer:

1. What semantic neighborhoods exist in the pooled book corpus?
2. What concepts and attributes overlap across those neighborhoods?
3. What does a particular reader's behavior say about each neighborhood?

Model C is intentionally richer than a single genre label or a single reader score.

The final Reading DNA representation keeps three reader signals separate:

- **Preference** — evidence that a reader likes a semantic neighborhood.
- **Avoidance** — evidence that a reader actively dislikes a semantic neighborhood.
- **Exploration** — evidence that a reader is actively exposed to a neighborhood without established preference.

A separate **evidence strength** measure describes how much actionable evidence supports the signal.

There is deliberately no single combined "Reading DNA score."

---

## Architecture

```text
                         CANONICAL BOOK CORPUS
                                  |
                                  v
                    Semantic representation
               title + author + description +
                    curated Open Library subjects
                                  |
                                  v
                    sentence embeddings
                    all-MiniLM-L6-v2 / 384d
                                  |
                                  v
                  weighted semantic book graph
                                  |
                                  v
                         Leiden communities
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
            Book neighborhoods          Book attributes
            "what belongs together?"   "what can overlap?"
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                       Reader book evidence
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
               preference     avoidance    exploration
                    |             |             |
                    +-------------+-------------+
                                  |
                                  v
                            Reading DNA
```

The semantic network is a pooled corpus representation. It is not fitted
separately for each reader, so community IDs remain comparable across the
three development readers.

---

## Semantic representation

The frozen semantic text contains:

- title
- author
- description
- curated Open Library subjects

Open Library subject metadata is normalized conservatively to remove obvious
catalog noise while preserving legitimate themes and concepts.

The embedding model is:

```text
all-MiniLM-L6-v2
384 dimensions
cosine-normalized
```

The development corpus contains 3,321 canonical books and 100% of those books
have semantic text.

The semantic representation is source-independent at the canonical layer:
Goodreads supplies reader/library evidence, while Open Library supplies book
metadata used for semantic representation.

---

## Semantic community detection

Flat KMeans was rejected as the final representation because it forces the
corpus into a small number of global partitions and does not represent the
fine-grained local structure observed in the semantic space.

HDBSCAN was also rejected for the current corpus because it produced a giant
central cluster and a large noise population.

The final semantic neighborhood representation is a weighted Leiden community
detection model over a thresholded k-nearest-neighbor graph.

### Frozen graph configuration

```text
Graph:
    weighted, symmetrized thresholded kNN graph

K:
    20 candidate neighbors per book

Similarity threshold:
    cosine similarity >= 0.40

Edge weight:
    cosine similarity

Forced weak edges:
    none

Community algorithm:
    Leiden

Partition:
    RBConfigurationVertexPartition

Resolution:
    3.0

Membership:
    books below the similarity threshold may remain isolated/unassigned
```

The graph was selected after comparing thresholded kNN configurations and
inspecting community size, coherence, stability, and representative books.

### Final development artifact

```text
Books:                 3,321
Communities:              55
Largest community:       257
Median community size:    52
Singleton communities:    12
```

Community IDs are implementation identifiers, not semantic labels. A future
production run should not treat an ID such as `community_id = 0` as a
permanent genre identity.

The current development baseline has been manually inspected and includes
coherent communities such as epic fantasy/romantasy, crime and psychological
suspense, sports romance, children's series, mythology romance, and other
fine-grained neighborhoods.

---

## Reader evidence

Reader evidence is attached at the `(reader_id, canonical_book_id)` level.

The reader-book evidence layer resolves repeated source records before
community-level aggregation.

### Preference signals

- `positive` = explicit positive preference, typically rating 3+ according
  to the canonical evidence semantics
- `negative` = explicit negative preference or DNF
- `read_without_preference` = evidence that the reader read the book without
  an explicit positive or negative preference
- `exposure_only` = TBR/current exposure without established preference
- `unknown` = insufficient evidence
- `conflicted` = conflicting evidence retained rather than silently resolved

The exact evidence semantics are defined in the canonical reader-evidence
documentation and are not duplicated as an independent source of truth here.

Unknown books do not contribute to preference, avoidance, or exploration.

---

# Reading DNA scoring contract

Reading DNA is calculated at the reader × community level.

## 1. Preference

Preference measures evidence that a reader likes a semantic neighborhood.

Let:

```text
P = positive_book_count
N = negative_book_count
E = P + N
```

Preference evidence strength:

```text
preference_evidence_strength =
    1 - exp(-E / 5)
```

A smoothed positive preference rate is then calculated using a prior strength
of 5 and prior preference of 0.25:

```text
smoothed_positive_rate =
    (P + 5 * 0.25)
    /
    (E + 5)
```

The final preference signal is:

```text
preference_strength =
    smoothed_positive_rate
    * preference_evidence_strength
```

The smoothing prevents a one-book community from appearing equally certain as
a community supported by dozens of books.

---

## 2. Avoidance

Avoidance is deliberately stricter than simply observing negative books.

A reader having many positive books and a few negative books in a community has
negative evidence, but that does not establish avoidance.

First calculate:

```text
positive_rate = P / (P + N)

negative_rate = N / (P + N)
```

Then:

```text
avoidance_balance =
    max(0, negative_rate - positive_rate)
```

Negative evidence strength is:

```text
negative_evidence_strength =
    1 - exp(-N / 5)
```

The avoidance signal is:

```text
avoidance_strength =
    avoidance_balance
    * negative_evidence_strength
```

A community must have at least **2 negative books** before it can appear in the
avoidance ranking.

This intentionally makes avoidance a high-evidence claim.

If no community qualifies, the correct output is:

```text
No qualifying communities.
```

The model does not manufacture avoidance merely to fill a dashboard slot.

---

## 3. Exploration

Exploration is not raw TBR volume.

It answers:

> "How much of this community represents active exposure without established
> preference?"

Let:

```text
X = exposure_only_book_count
A = preference_evidence_count + X
```

Then:

```text
exploration_rate =
    X / A
```

Exploration evidence strength:

```text
exploration_evidence_strength =
    1 - exp(-X / 5)
```

Final exploration signal:

```text
exploration_strength =
    exploration_rate
    * exploration_evidence_strength
```

This allows two different situations to be distinguished:

```text
Established interest + continued exploration
```

versus:

```text
Primarily exploratory exposure
```

For example, in the current development reader data, one community has
43 positive preference books and 90 exposure-only books. Its exploration
signal is therefore meaningful but below a community with 13 exposure-only
books and no established preference evidence.

---

## 4. Evidence strength

Evidence strength is deliberately separate from signal direction.

```text
actionable_evidence =
    preference_evidence_count
    + exposure_only_book_count
```

Then:

```text
evidence_strength =
    1 - exp(-actionable_evidence / 5)
```

This answers:

> "How much evidence do we have?"

It does not answer:

> "Does the reader like it?"

---

# Current Reading DNA experiment

The scoring experiment was run against:

```text
Readers:       3
Communities:  55
Reader/community rows: 132
```

The three readers are the controlled development validation set:

- You
- Sarah
- Shannon

This is **not** generalization evidence. The three readers are useful for
controlled model development, but broader validation with more diverse reader
profiles is still required.

## Results

### Preference

All three readers produced interpretable preference rankings supported by
substantial evidence.

For the current user:

```text
Community 0:
    preference = 0.9217
    positive = 43
    negative = 0
    preference evidence = 43

Community 11:
    preference = 0.6252
    positive = 10
    negative = 1
    preference evidence = 11

Community 8:
    preference = 0.5180
    positive = 7
    negative = 0
    preference evidence = 7
```

These results are consistent with the intended behavior: large, explicitly
positive evidence produces stronger and more stable preference signals.

### Avoidance

No community qualified for avoidance for any of the three readers.

Validation:

```text
Avoidance validation: PASS
Zero-negative communities with positive avoidance score: 0
Zero-negative communities ranked for avoidance: 0
```

This is an acceptable result. The current sample does not provide strong
enough community-level evidence to claim that any reader actively avoids a
semantic neighborhood.

### Exploration

The exploration signal successfully identifies communities where exposure
exists without established preference.

For the current user:

```text
Community 28:
    exploration = 0.9257
    preference evidence = 0
    exposure-only = 13

Community 9:
    exploration = 0.7761
    preference evidence = 6
    exposure-only = 22

Community 0:
    exploration = 0.6767
    preference evidence = 43
    exposure-only = 90
```

This demonstrates an important distinction for the eventual product:

A reader can have an **established interest** in a neighborhood while also
**continuing to explore** within it.

---

# Why the signals remain separate

A single weighted Reading DNA score would hide important differences.

For example:

```text
High preference + high exploration
```

means:

> "I strongly like this area and am still exploring it."

Whereas:

```text
Low preference evidence + high exploration
```

means:

> "I'm actively exploring this area, but we don't yet know whether I like it."

And:

```text
High avoidance
```

means something materially different again:

> "There is enough negative evidence that this neighborhood may be a poor fit."

These should remain distinct features of the reader profile.

---

# Current status

### Frozen

- canonical reader/book evidence
- reader/book evidence aggregation
- semantic representation
- sentence embedding model
- weighted thresholded kNN graph
- Leiden community detection baseline
- Reading DNA preference/avoidance/exploration scoring contract

### Still experimental

- production Reading DNA data contract
- reader-level narrative/profile generation
- multi-label attribute integration into the final profile
- recommendation integration
- validation across a broader and more diverse reader population

### Not yet production

The experiment outputs are analytical artifacts. The scoring module should not
yet be treated as the final public-user pipeline.

The next step is to build the **reader-level Reading DNA representation** from
these validated community signals.

---

## Experiment artifacts

```text
data/processed/canonical/reading_dna/
├── reader_community_evidence.csv
├── reader_unassigned_evidence.csv
├── reader_community_reading_dna_experiment.csv
└── reader_community_reading_dna_rankings.csv
```

The experiment runner is:

```powershell
python scripts/run_reader_community_scoring.py
```

The output should be inspected before promotion into production analytics.

---

## Model relationship

```text
Model A
    overall positive-preference representation
    "What do I like overall?"

Model B
    multiple semantic preference neighborhoods
    "What distinct areas of my taste exist?"

Model C
    semantic communities
    + multi-label attributes
    + reader evidence
    + preference / avoidance / exploration
    "How does this reader actually read?"
```

Model C is the richer representation layer that can later support:

- personalized recommendations
- Reader Match
- Book Match
- evidence-based "Why this book?" explanations
- shareable Reading DNA
- the Streamlit Reading DNA experience
