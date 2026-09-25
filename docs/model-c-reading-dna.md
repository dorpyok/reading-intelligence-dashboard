# Model C — Multi-Interest + Multi-Label Reading DNA

## Purpose

Model C is the production candidate for the Reading Intelligence Dashboard's
Reading DNA layer.

It answers three different questions without collapsing them into one score:

1. **Interest clusters:** What semantic neighborhoods exist in the reader's
   book corpus?
2. **Attributes/themes:** What concepts describe books, including concepts
   that can overlap across multiple clusters?
3. **Evidence:** What does the reader's history say about each attribute and
   cluster in terms of preference, exposure, and reading intent?

A book can therefore belong to several attributes at once.

Example:

> a book could carry `horror`, `feminist fiction`, and `speculative fiction`
> simultaneously.

## Model C architecture

```text
                         BOOK CORPUS
                              |
                       sentence embeddings
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
          INTEREST CLUSTERS          MULTI-LABEL ATTRIBUTES
                 |                          |
       "what belongs together?"     "what can overlap?"
                 |                          |
                 +------------+-------------+
                              |
                         READER EVIDENCE
                    /           |           \
              preference     exposure      intent
                    \           |           /
                     +----------+----------+
                              |
                         READING DNA
```

## Clustering design

The model uses a pooled book corpus rather than fitting a separate cluster
number for each reader. This makes cluster IDs comparable across readers.

KMeans is retained for V1 because:

- Model B already established a KMeans baseline.
- It produces a complete assignment rather than a large outlier bucket.
- It is simple to fit and reproduce.
- Model C uses clusters as discovery/navigation structure, not as
  authoritative genre labels.

The final K is selected by evaluating a range of values and applying explicit
guardrails:

1. stability across random seeds must be at least 0.70 ARI;
2. the smallest cluster must contain at least 1% of the pooled corpus;
3. among eligible solutions, choose the highest mean silhouette;
4. ties go to the smaller K;
5. if no candidate meets the guardrails, fall back to highest silhouette and
   preserve that fact in the tuning output.

This is intentionally not a single hidden weighted objective.

## Multi-label attribute design

Attributes come from Open Library `subjects`, normalized for:

- case
- punctuation
- whitespace
- obvious metadata terms
- obvious location/catalog noise

The model does **not** force each book into one genre.

Attributes remain evidence-backed metadata. Model C does not invent a taxonomy
or ask an LLM to assign labels during the core analytical pipeline.

## Reader evidence

Each attribute is measured independently through:

### Preference

- positive: rating >= 4
- negative: rating <= 2 or did not finish
- neutral: rating == 3 or no explicit preference evidence

### Exposure

Observed reading:

- read
- did not finish
- currently reading

### Intent

Current V1 intent:

- to-read

These streams remain separate. Model C does not apply arbitrary weights.

## Outputs

```text
data/processed/reading_dna/
├── book_reading_dna.csv
├── cluster_descriptions.csv
├── cluster_tuning.csv
├── reader_attributes.csv
├── reader_attribute_combinations.csv
├── reader_clusters.csv
├── model_c_summary.csv
└── model_c_config.json
```

## Run

From the repository root:

```powershell
python scripts/run_model_c_reading_dna.py --reader all
```

Run one reader while debugging:

```powershell
python scripts/run_model_c_reading_dna.py --reader you
```

Run tests:

```powershell
pytest -q
```

## What counts as a final model?

Do not choose the final Model C configuration from silhouette alone.

After the tuning run, inspect:

- cluster stability
- minimum cluster size
- cluster sizes
- top attributes by cluster
- reader cluster coverage
- attribute coverage
- whether known multi-label examples behave sensibly
- whether the resulting Reading DNA is interpretable

The tuning CSV is therefore part of the model artifact, not disposable output.

## Relationship to Model A and Model B

```text
Model A
single positive-preference centroid
"What do I like overall?"

Model B
multiple positive-preference centroids
"What distinct semantic neighborhoods exist inside what I like?"

Model C
clusters + overlapping attributes + evidence streams
"What are my interest neighborhoods, what concepts cross them,
and what does my behavior say about each?"
```

Model C does not replace the Model B recommendation experiment. Model B remains
the recommendation-oriented benchmark. Model C is the richer representation
layer that can later feed recommendations, Reader Match, Book Match, AI
explanations, and the Streamlit Reading DNA experience.
