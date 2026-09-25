# Reading Intelligence Dashboard
## Requirements, Goals, Scope & Success Criteria

**Document status:** Living project requirements  
**Current phase:** Model C / Reading DNA  
**Primary product:** Free public interactive Reading Intelligence tool  
**Repository:** `reading-intelligence-dashboard`

---

# 1. Product Vision

Build a reusable, public Reading Intelligence tool that turns a reader's
existing reading history into an understandable picture of how they read.

The product should go beyond a generic "book recommendation" engine.

It should help a reader understand:

- what they tend to read;
- what they tend to enjoy;
- what themes and attributes cross their reading interests;
- what they have explored versus what they appear to prefer;
- what they are currently interested in;
- which books they may enjoy next;
- how their reading profile compares with another reader;
- why a recommendation or match was made.

The central product concept is **Reading DNA**.

---

# 2. Product End Goals

## 2.1 Reading DNA

The tool should create a reader-level representation that combines:

- semantic interest neighborhoods;
- overlapping book attributes/themes;
- explicit preference evidence;
- reading exposure/behavior;
- prospective reading intent.

Reading DNA should be interpretable. It should not reduce a reader to
one opaque score or one genre label.

## 2.2 Personalized Recommendations

The product should eventually recommend books using the reader's learned
representation rather than only simple genre matching.

Recommendations should be explainable.

The product should be able to answer a question such as:

> "Why does this book look like a fit for me?"

The recommendation layer should remain distinct from the Reading DNA
representation layer.

## 2.3 Reader Match

Allow two readers to compare their reading data.

Potential capabilities include:

- books they have both read;
- books one reader has read and the other has not;
- shared interests;
- overlapping attributes;
- areas where their reading differs;
- books they might consider reading together.

Reader Match should use the same canonical reader/book representations as the
individual reader experience rather than creating a second disconnected model.

## 2.4 Book Match

Given a book, explain how it relates to the reader's Reading DNA.

Potential questions:

- Which interests does it touch?
- Which attributes overlap with the reader's preferences?
- Is it similar to books the reader rated highly?
- Is it novel territory for the reader?
- Does it overlap with books already on the reader's TBR?

## 2.5 Shareable Outputs

The product should eventually produce attractive, understandable outputs that
a reader can share.

Examples:

- Reading DNA card;
- favorite-interest summary;
- recommendation card;
- Reader Match summary;
- Book Match summary;
- Instagram-ready visual/content output.

These should be generated from analytical outputs rather than manually
constructed for individual examples.

---

# 3. Product Constraints

## Required

- Free public access.
- No paid subscription required.
- No freemium gating.
- No user account required for the core experience.
- No payment system required for the core experience.
- Must be reusable for multiple readers.
- Must preserve source data rather than silently discarding uncertain records.
- Must distinguish observed evidence from inference.
- Must keep analytical/modeling logic separate from AI-generated explanations.
- Must be understandable to a nontechnical reader.

## Optional / later

- Voluntary Support/Donate mechanism.
- Additional reading-data sources.
- Additional export/share formats.
- More sophisticated AI explanations.
- Persistent user accounts, only if a future product decision establishes a
  real need.

These optional capabilities must not become dependencies of the core product.

---

# 4. Data Source Requirements

## V1 source

Goodreads public reading data is the initial source.

The current ingestion approach uses public Goodreads RSS rather than the
retired/deprecated Goodreads public API.

The ingestion layer must isolate Goodreads-specific behavior so the canonical
model does not depend on Goodreads terminology.

## Future source independence

The canonical model must support additional sources without requiring the
analytics and recommendation layers to be rewritten.

Source-specific complexity belongs at the ingestion boundary.

---

# 5. Canonical Data Model Requirements

The canonical model is intentionally source-independent.

Core entities:

```text
Reader
   │
   └── Reading Record ─── Book
             │
             └── Shelf
```

The current canonical database contains:

- `reader`
- `book`
- `reading_record`
- `shelf`

The reading record stores:

- reader;
- book;
- reading status;
- rating;
- date added;
- date read;
- review;
- source;
- source record ID;
- ingestion timestamp.

The book stores source-independent book information plus the Goodreads source
identifier where available.

---

# 6. Reading Status Requirements

Canonical reading status values:

- `to_read`
- `currently_reading`
- `read`
- `did_not_finish`
- `unknown`

Status derivation must use explicit evidence precedence.

Current intended precedence:

```text
did_not_finish
      ↓
currently_reading
      ↓
to_read
      ↓
date_read
      ↓
rating > 0
      ↓
unknown
```

Important rules:

- `date_added` is not evidence that a book was read.
- A blank shelf is not evidence that a book was read.
- A rating greater than zero is strong evidence of having read the book.
- Goodreads rating `0` means unrated and should become canonical NULL.
- Raw source values must remain available.
- Conflicting evidence should not be silently resolved by guessing.

---

# 7. Data Quality Requirements

The product must retain imported books even when reading evidence is
incomplete.

Books without sufficient evidence should become:

**Needs confirmation**

rather than being deleted or automatically classified as read.

The user should be able to understand why a record needs confirmation and what
could strengthen the data, such as:

- adding a rating;
- adding a read date;
- correcting reading status.

The product may optionally identify books marked/read without a review and
suggest adding a review, but reviews are not required for inclusion.

---

# 8. Enrichment Requirements

Open Library is the current enrichment source.

Enrichment may provide:

- subjects;
- authors;
- work/edition metadata;
- publication information;
- identifiers;
- descriptions/covers where available.

Enrichment must be treated as supporting metadata, not unquestioned truth.

The pipeline should preserve:

- match method;
- match status;
- source identifiers;
- raw enrichment information where useful;
- unmatched records.

The system must tolerate incomplete enrichment.

---

# 9. Semantic Representation Requirements

The current semantic representation uses:

**Sentence Transformers — `all-MiniLM-L6-v2`**

Book representations incorporate:

- title;
- author;
- Open Library subjects;
- cleaned description.

Text normalization must handle:

- HTML;
- HTML entities;
- common encoding/mojibake problems;
- whitespace normalization.

The same canonical text representation should be reused across experiments
unless a documented experiment explicitly tests another representation.

---

# 10. Model Progression

The three-model progression is intentional.

## Model A — Single Preference Centroid

Question:

> What do I like overall?

Representation:

- explicit positive preference;
- one semantic centroid.

Purpose:

- establish the simplest recommendation baseline.

---

## Model B — Multi-Interest Preference

Question:

> What distinct semantic neighborhoods exist within what I like?

Representation:

- positive books;
- multiple KMeans centroids;
- candidate score based on similarity to the closest preference centroid.

Purpose:

- test whether one preference centroid is too restrictive.

Model B remains an important **recommendation benchmark**.

It should not be discarded simply because Model C is richer.

---

## Model C — Multi-Interest + Multi-Label Reading DNA

Question:

> What interest neighborhoods exist, what attributes cross those
> neighborhoods, and what does my reading evidence say about each?

Model C contains two complementary analytical layers:

### A. Interest clustering

Answers:

> What belongs together semantically?

Clusters are a discovery/navigation mechanism.

They are **not** authoritative genre labels.

### B. Multi-label attributes

Answers:

> What concepts can overlap?

A book can have multiple attributes simultaneously.

For example, a book may be:

```text
horror
feminist fiction
speculative fiction
```

It must not be forced into only one category.

### C. Reader evidence

Each attribute and cluster can be interpreted through separate evidence
streams:

```text
Preference
Exposure
Intent
```

These streams should initially remain separate rather than being collapsed
into arbitrary weights.

---

# 11. Model C Requirements

## Clustering

The final cluster configuration must be selected empirically.

Selection should consider:

- silhouette;
- stability;
- cluster size;
- semantic interpretability;
- reader coverage.

Silhouette alone is insufficient.

The final model must preserve its tuning results as an artifact.

## Multi-label attributes

Attributes should be:

- evidence-backed;
- reproducible;
- normalized;
- multi-label;
- allowed to cross cluster boundaries.

The first implementation should use Open Library subject evidence rather than
inventing a large manual taxonomy.

## Reader-level representation

For every attribute, the model should be able to distinguish:

- positive preference evidence;
- negative preference evidence;
- neutral/no explicit preference;
- observed exposure;
- prospective intent.

No opaque combined score should be required for the foundational model.

---

# 12. Recommendation Requirements

Recommendations must eventually use the Reading DNA representation.

The system should support:

1. candidate generation;
2. similarity/relevance scoring;
3. exclusion of books the reader has already read where appropriate;
4. preference-aware ranking;
5. explanation of why a book was recommended.

Recommendation evaluation must distinguish between:

- known-book retrieval experiments;
- genuinely unseen-book recommendation;
- user-facing recommendation quality.

The existing Model A/B experiments are baselines, not proof that the product
will recommend genuinely new books well.

---

# 13. Reader Match Requirements

Reader Match should use shared canonical representations.

Required conceptual outputs:

- common books;
- shared attributes/interests;
- differences in reading interests;
- potentially complementary reading suggestions.

A Reader Match must not simply calculate overlap of raw Goodreads shelves.

It should use the analytical representations developed for Reading DNA.

---

# 14. Book Match Requirements

For a selected book, Book Match should explain:

- semantic similarity to the reader's interests;
- matching attributes;
- relationship to highly rated books;
- whether the book represents familiar or novel territory;
- relationship to TBR/intent where applicable.

The output should be understandable without exposing model internals.

---

# 15. AI Requirements

AI is an **interpretation/explanation layer**, not the source of truth for
the analytical model.

The system should follow:

```text
raw data
   ↓
canonical data
   ↓
enrichment
   ↓
analytics / ML
   ↓
Reading DNA / recommendation results
   ↓
AI explanation
```

AI should explain computed evidence rather than inventing unsupported
preferences.

Examples:

- explain why a book matched;
- summarize Reading DNA;
- describe a Reader Match;
- generate shareable text.

AI-generated language must remain grounded in structured analytical outputs.

---

# 16. Application Requirements

Target application:

**Streamlit**

The application should eventually expose:

### Reader onboarding
- provide/import reading data;
- validate data;
- explain data quality.

### Reading DNA
- interest clusters;
- overlapping attributes;
- preference;
- exposure;
- intent;
- understandable explanations.

### Recommendations
- personalized books;
- reasons for recommendations.

### Reader Match
- compare two readers.

### Book Match
- evaluate a book against a reader.

### Sharing
- create shareable summaries/cards.

The analytical code must remain usable independently of Streamlit.

---

# 17. Reusability Requirements

The project is a reusable product, not a one-off analysis of the owner's
Goodreads library.

The pipeline must support:

- multiple readers;
- different library sizes;
- incomplete metadata;
- different preference distributions;
- multiple source users;
- future source adapters.

Reader-specific assumptions must not be embedded into production analytics.

The current three-reader validation corpus is useful for model development,
but it is not the definition of the product.

---

# 18. Engineering Requirements

The project should demonstrate professional engineering practices.

Required:

- modular Python source;
- tests for analytical logic;
- reproducible experiments;
- explicit model configuration;
- separation of raw/canonical/enriched/analytical layers;
- documented architectural decisions;
- deterministic random seeds where appropriate;
- clear experiment outputs;
- validation before promoting an experiment into production;
- no unexplained hard-coded model weights.

The project should favor the simplest design that adequately solves the
problem.

Do not over-engineer V1.

---

# 19. Current Architecture

Conceptually:

```text
GOODREADS / FUTURE SOURCES
            ↓
        INGESTION
            ↓
          RAW
            ↓
        CANONICAL
            ↓
       ENRICHMENT
            ↓
        ANALYTICS
       /    |     \
      /     |      \
   Model A Model B Model C
                     |
               Reading DNA
                     ↓
             RECOMMENDATION
                     ↓
                  AI
                     ↓
               STREAMLIT
                     ↓
              PUBLIC PRODUCT
```

The architecture must preserve the distinction between:

- source ingestion;
- canonical data;
- enrichment;
- analytical models;
- recommendations;
- AI;
- application/UI.

---

# 20. Portfolio / Career Goal

This project is also a portfolio demonstration.

It should demonstrate that the builder can:

- define a business problem;
- translate it into measurable analytical requirements;
- design a canonical data model;
- build reusable ingestion;
- perform data enrichment;
- engineer semantic representations;
- experiment with ML approaches;
- evaluate competing models;
- avoid overclaiming from weak validation;
- build interpretable analytical products;
- separate analytical truth from generative AI;
- build a usable public application.

The project should demonstrate **decision science**, not merely model
building.

The important portfolio story is:

> The model was not selected because it was the fanciest model. It was
> developed through successive hypotheses, evaluated against real reader
> behavior, and selected based on evidence and product requirements.

---

# 21. Explicit Non-Goals

These are intentionally outside V1.

## Not required now

- user accounts;
- subscriptions;
- freemium tiers;
- payment processing;
- enterprise-scale infrastructure;
- production vector database;
- a giant manually curated genre ontology;
- LLM-generated classification as the analytical foundation;
- perfect Goodreads edition/work identity resolution;
- replacing the canonical model with an embedding store;
- building a social network;
- building a full publishing analytics platform inside this project.

Those may be appropriate for the separate Book & Culinary Intelligence
Platform or future product phases.

---

# 22. Model C Definition of Done

Model C is not complete merely because the script runs.

It is complete when:

- [ ] semantic clustering is implemented;
- [ ] cluster count has been tuned;
- [ ] cluster stability has been measured;
- [ ] cluster-size guardrails have been evaluated;
- [ ] final K has been documented;
- [ ] cluster descriptions are interpretable;
- [ ] multi-label attributes are extracted;
- [ ] attribute normalization is tested;
- [ ] attributes can cross cluster boundaries;
- [ ] preference evidence is represented;
- [ ] exposure evidence is represented;
- [ ] intent evidence is represented;
- [ ] evidence streams remain separate;
- [ ] reader-level attributes are generated;
- [ ] attribute combinations are generated;
- [ ] three-reader validation has been run;
- [ ] results have been inspected for semantic sanity;
- [ ] tests pass;
- [ ] production source is moved into `src/analytics`;
- [ ] model configuration is reproducible;
- [ ] Model B remains available as a comparison baseline;
- [ ] Model C documentation is updated.

---

# 23. Reading DNA Product Definition of Done

The Reading DNA feature is complete when a reader can see:

- [ ] their major semantic interest areas;
- [ ] the attributes that characterize those areas;
- [ ] attributes that cross multiple areas;
- [ ] evidence of what they tend to enjoy;
- [ ] evidence of what they have explored;
- [ ] evidence of what they currently intend to read;
- [ ] appropriate uncertainty where evidence is weak;
- [ ] understandable explanations;
- [ ] no misleading single-number "personality" score.

---

# 24. Public Product Definition of Done

The public V1 product is complete when a new user can:

- [ ] provide supported reading data;
- [ ] receive a validated reading library;
- [ ] understand data-quality issues;
- [ ] receive a Reading DNA profile;
- [ ] receive recommendations;
- [ ] understand recommendation explanations;
- [ ] compare with another reader;
- [ ] perform Book Match;
- [ ] create a shareable result;
- [ ] use the tool without a paid subscription;
- [ ] use the tool without requiring an account.

---

# 25. Current Status

## Completed

- [x] Repository established.
- [x] Source-independent canonical model established.
- [x] Goodreads RSS ingestion validated with multiple readers.
- [x] Reading status derivation established.
- [x] Data-quality handling established.
- [x] Open Library enrichment pipeline established.
- [x] Metadata normalization established.
- [x] Semantic book representation established.
- [x] Model A baseline established.
- [x] Model B baseline established.
- [x] Model B sensitivity analysis completed.
- [x] Reader preference structure analysis completed.
- [x] Reader profile interpretation layer prototyped.
- [x] Model C conceptual architecture established.
- [x] Initial Model C implementation created.
- [x] Initial Model C tests created.

## Current

**Model C empirical tuning and validation**

The next decision is not "does Model C run?"

The next decision is:

> Does the resulting combination of clusters + multi-label attributes +
> preference/exposure/intent evidence produce a Reading DNA representation
> that is coherent, useful, reproducible, and better aligned with the product
> goals than the simpler alternatives?

## Next

1. Run the Model C test suite in the repository.
2. Run Model C against the three validation readers.
3. Inspect cluster tuning.
4. Inspect cluster descriptions.
5. Inspect attribute coverage.
6. Inspect reader attribute profiles.
7. Inspect attribute combinations.
8. Compare Model B and Model C where a common evaluation is meaningful.
9. Fine-tune Model C.
10. Promote the final implementation into `src/analytics`.
11. Freeze the Model C baseline.
12. Build the Reading DNA application layer.

---

# 26. Model Evaluation Principle

The project should not ask:

> "Which model has the highest metric?"

It should ask:

> "Which model best satisfies the analytical and product requirements while
> remaining interpretable, reproducible, and useful?"

Different models answer different questions.

Therefore:

```text
Model A → baseline simplicity
Model B → recommendation-oriented multi-interest retrieval
Model C → Reading DNA representation
```

Model C does not need to beat Model B on every recommendation metric to be the
correct Reading DNA architecture.

Conversely, Model C should not be promoted merely because it is more complex.

The final decision must be evidence-based.

---

# 27. Future Expansion

Potential future capabilities:

- additional reading sources;
- richer book/work/edition identity;
- better subject normalization;
- learned attribute taxonomy;
- stronger unseen-book recommendation evaluation;
- graph-based reader/book relationships;
- richer Reader Match;
- richer Book Match;
- AI-generated Reading DNA narratives;
- Instagram/social content generation;
- voluntary support;
- additional public analytical views.

Future work should be evaluated against this requirements document before
being added to the core product.

---

# 28. Guiding Principle

The product should make a reader feel:

> "This actually describes how I read."

Not:

> "An algorithm put me into a genre."

The model is successful when it captures the **structure of a reader's
interests and reading behavior**, including overlap, ambiguity, exploration,
and change.
