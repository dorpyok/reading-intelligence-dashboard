# Reading Intelligence Architecture

## Purpose

Reading Intelligence is a reusable reading analytics and recommendation
application that allows users to provide their reading data and explore
personalized insights.

The application is designed to remain lightweight while maintaining clear
separation between ingestion, canonical data, enrichment, analytics,
recommendations, and presentation.

## Architecture

```text
User Data
   │
   ▼
Ingestion
   │
   ▼
Raw Data
   │
   ▼
Canonical Data Model
   │
   ├───────────────┐
   ▼               ▼
Enrichment      Analytics
   │               │
   └───────┬───────┘
           ▼
    Recommendation
       Engine
           │
           ▼
     AI Explanations
           │
           ▼
      Streamlit App
           │
           ▼
       Public Tool

## Machine Learning and Semantic Intelligence

The Reading Intelligence Dashboard uses a staged ML architecture.

External reading data is first ingested and transformed into the canonical reading model. Book records can then be enriched with external bibliographic metadata before semantic feature engineering.

The current exploratory ML pipeline uses:

- Sentence Transformers for semantic embeddings
- UMAP for dimensionality reduction
- HDBSCAN for unsupervised clustering
- TF-IDF for cluster descriptor terms

The current experiment compares a Goodreads-only baseline representation with a Goodreads + Open Library enriched representation.

The clustering layer is intentionally separate from recommendation logic. Clusters describe naturally occurring neighborhoods within a reader's library, while semantic similarity can later support book-to-book recommendations.

Future supervised modeling may incorporate reader ratings and reading behavior to learn individual preferences.

See `docs/ml-clustering-experiment.md` for experiment results and limitations.

Current Data Flow 
Goodreads
   ↓
Ingestion
   ↓
Raw reading data
   ↓
Canonical reading model
   ↓
Open Library enrichment
   ↓
Book representation
   ↓
Analytics / ML
   ↓
Reader preference representation
   ↓
Recommendations
   ↓
Streamlit application