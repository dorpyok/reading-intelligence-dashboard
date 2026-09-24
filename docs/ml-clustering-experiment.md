# ML Clustering Experiment

## Purpose

This experiment tests whether enriching Goodreads reading data with external Open Library bibliographic metadata improves the semantic representation of a reader's book collection.

The objective is not to force books into conventional genre categories.

Instead, the experiment uses unsupervised machine learning to identify naturally occurring semantic neighborhoods across the reader's library.

---

## Experiment Scope

Dataset:

- 359 Goodreads books
- Single-reader library
- Goodreads data used as the source reading history
- Open Library used as an external metadata enrichment source

The experiment compares two representations of the same 359 books.

### Model A — Goodreads Baseline

The baseline representation uses:

- title
- author
- Goodreads description

Pipeline:

Goodreads → Text representation → Sentence Transformer embeddings → UMAP dimensionality reduction → HDBSCAN clustering

### Model B — Enriched Representation

The enriched representation adds Open Library subject metadata to the Goodreads representation.

Pipeline:

Goodreads + Open Library enrichment → Enriched text representation → Sentence Transformer embeddings → UMAP dimensionality reduction → HDBSCAN clustering

---

## Semantic Representation

Books are represented using the `all-MiniLM-L6-v2` Sentence Transformer model.

The representation combines:

- title
- author
- subjects
- description

The model produces a 384-dimensional semantic embedding for each book.

The pretrained Sentence Transformer is not fine-tuned on the reader's library.

The embeddings provide a semantic representation of each book, while UMAP and HDBSCAN are fitted to the reader's dataset.

---

## Unsupervised Learning Approach

### UMAP

UMAP is used for two purposes:

1. Reduce the embeddings to a lower-dimensional representation suitable for clustering.
2. Create a two-dimensional visualization of the semantic book space.

The clustering representation uses 10 dimensions.

The visualization uses a separate two-dimensional UMAP projection.

The 2D visualization should therefore be treated as an interpretation aid rather than the actual clustering space.

### HDBSCAN

HDBSCAN is used to identify dense groups of semantically similar books.

HDBSCAN was selected because it:

- does not require a predefined number of clusters
- can identify clusters with different densities
- can classify books as outliers
- provides cluster membership probabilities

An outlier is represented by `cluster_id = -1`.

An outlier does not necessarily indicate bad data. It indicates that the algorithm did not find sufficient evidence that the book belongs to a stable dense cluster under the current representation and parameters.

---

# Model Comparison

Both models were run against the same 359-book library.

| Metric | Model A: Goodreads Baseline | Model B: Open Library Enriched |
|---|---:|---:|
| Books | 359 | 359 |
| Discovered clusters | 14 | 15 |
| Books assigned to clusters | 294 | 242 |
| HDBSCAN outliers | 65 (18.1%) | 117 (32.6%) |
| Largest cluster | 110 books | 30 books |
| Median cluster size among assigned books | 11 | 13 |
| Average cluster membership probability | 0.923 | 0.912 |
| Normalized cluster-size entropy | 0.818 | 0.960 |

## Observed Differences

### Cluster differentiation

The baseline model produced one very large cluster containing 110 books.

That cluster included heterogeneous examples spanning literary fiction, thrillers, speculative fiction, classics, and other categories.

The enriched model produced 15 clusters and reduced the largest cluster to 30 books.

This indicates that the additional Open Library metadata substantially changed the organization of the semantic book space and allowed the clustering algorithm to distinguish more localized neighborhoods.

### Outlier behavior

The proportion of books classified as outliers increased:

- Model A: 65 / 359 = 18.1%
- Model B: 117 / 359 = 32.6%

The enriched model therefore became more selective about cluster membership.

This should not automatically be interpreted as either positive or negative.

Possible interpretations include:

- external metadata provides enough additional information to distinguish books that previously appeared similar
- some books genuinely occupy unusual semantic positions within the reader's collection
- the representation may contain heterogeneous or noisy subject metadata
- HDBSCAN parameters may require later evaluation

Further validation is required before using outlier rate as a model-selection criterion.

### Cluster confidence

Average membership probability among assigned books remained high:

- Model A: 0.923
- Model B: 0.912

The enriched model therefore did not simply replace the large cluster with uniformly low-confidence assignments.

Instead, it assigned fewer books to clusters while maintaining relatively high membership confidence for books that were assigned.

### Cluster-size distribution

Normalized cluster-size entropy increased from:

- Model A: 0.818
- Model B: 0.960

This indicates a more evenly distributed cluster structure in the enriched model.

The result is consistent with the observed reduction of the large baseline catch-all cluster.

---

# Interpretation

The experiment demonstrates that external bibliographic metadata can materially change the semantic structure discovered within a reader's collection.

The enriched model produced more differentiated semantic neighborhoods rather than concentrating a large portion of the library into one broad cluster.

Examples of observed semantic neighborhoods included:

- epic/high fantasy series
- romantasy and fantasy series
- dark/paranormal/witchy fantasy
- fairy-tale and folkloric fantasy
- science fiction and speculative fiction
- literary and contemporary fiction
- romance and domestic/thriller crossover

These should be interpreted as discovered semantic neighborhoods rather than authoritative genre classifications.

The model is intentionally not restricted to a predefined taxonomy.

---

# Important Modeling Distinction

Cluster membership and book similarity are related but different concepts.

### Clustering

Answers:

> What naturally occurring groups exist in this reader's library?

### Semantic similarity

Answers:

> Which individual books are most similar to this book?

The eventual recommendation system should not rely exclusively on cluster membership.

A future recommendation architecture may use the underlying embeddings to calculate book-to-book similarity while using clusters as an additional representation of the reader's reading universe.

---

# Current Limitations

Several limitations remain.

### 1. Small dataset

The experiment contains 359 books from one reader.

The results should therefore be treated as an ML experiment rather than evidence of general performance across readers.

### 2. Open Library subject quality

Open Library subjects are heterogeneous.

They can contain:

- genres
- themes
- settings
- historical contexts
- awards
- series metadata
- publishing metadata
- audience classifications
- cataloging terminology

The current experiment intentionally preserves these signals rather than imposing a manual taxonomy.

Future experiments should evaluate whether some subject categories should receive different treatment.

### 3. Embedding model

`all-MiniLM-L6-v2` was selected as a lightweight baseline appropriate for the current dataset size.

Other embedding models may produce different semantic neighborhoods.

### 4. UMAP sensitivity

UMAP projections can change with parameters and random seeds.

The 2D visualization should therefore not be treated as a definitive map of semantic distance.

### 5. HDBSCAN sensitivity

Cluster membership depends on parameters including:

- `min_cluster_size`
- `min_samples`
- dimensionality reduction parameters
- embedding representation

The current parameters are experimental rather than optimized.

---

# Why Model B Was Retained

Model B was retained as the current enriched experiment because it demonstrated substantially greater differentiation of the book collection.

However, no claim is made that Model B is universally superior to Model A.

The next evaluation stage should determine whether the additional structure is useful for downstream tasks such as:

- book-to-book similarity
- reader preference modeling
- recommendation generation
- Reader DNA
- explainable recommendations

---

# Portfolio / Resume Significance

This experiment demonstrates an end-to-end unsupervised ML workflow:

1. Ingested messy external reading data.
2. Preserved the raw source representation.
3. Enriched records using an external bibliographic data source.
4. Engineered semantic text representations.
5. Generated transformer-based embeddings.
6. Applied dimensionality reduction.
7. Applied density-based unsupervised clustering.
8. Generated cluster descriptors using TF-IDF.
9. Compared a baseline representation with an enriched representation.
10. Evaluated changes in cluster structure, outlier rate, membership confidence, and cluster-size distribution.
11. Created a visualization of the resulting semantic book space.

This establishes the foundation for later recommendation and reader-preference modeling.

---

# Experiment Status

**Status:** Complete — exploratory ML milestone

**Current model:** Goodreads + Open Library enriched semantic representation

**Next ML phase:** Reader preference modeling and book similarity/recommendation

No further clustering optimization is required for this milestone.