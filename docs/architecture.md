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