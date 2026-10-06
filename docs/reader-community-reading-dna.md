# Reader × Community Reading DNA Scoring Experiment

## Purpose

This experiment converts aggregated reader/book evidence into descriptive
signals at the intersection of:

```text
reader × semantic community
```

The experiment is intentionally separate from the final product-level Reading
DNA representation.

Its purpose is to determine whether the evidence produces useful,
interpretable signals before those signals are promoted into the application.

---

## Inputs

Reader/book evidence:

```text
data/processed/canonical/reader_book_evidence.csv
```

Final semantic community assignments:

```text
data/processed/canonical/leiden/leiden_assignments.csv
```

The experiment produces:

```text
data/processed/canonical/reading_dna/
├── reader_community_evidence.csv
├── reader_unassigned_evidence.csv
├── reader_community_reading_dna_experiment.csv
└── reader_community_reading_dna_rankings.csv
```

---

## Development population

The current controlled validation set contains:

```text
Readers:       3
Communities:  55
Reader/community observations: 132
```

The three readers are useful for controlled development and manual sanity
checking.

They are **not sufficient to establish generalization** across different
reading profiles.

---

# Signal design

The experiment deliberately rejects a single composite Reading DNA score.

There are four separate concepts:

1. preference strength
2. avoidance strength
3. exploration strength
4. evidence strength

---

## Preference strength

Preference uses explicit positive and negative preference evidence.

```text
P = positive books
N = negative books
E = P + N
```

Preference evidence strength:

```text
1 - exp(-E / 5)
```

Smoothed positive rate:

```text
(P + 5 × 0.25) / (E + 5)
```

Preference strength:

```text
smoothed_positive_rate
×
preference_evidence_strength
```

The smoothing prevents tiny samples from receiving unjustified certainty.

---

## Avoidance strength

Avoidance requires stronger evidence than simply having a few negative books.

```text
positive_rate = P / (P + N)

negative_rate = N / (P + N)
```

Only the portion where negative evidence exceeds positive evidence contributes:

```text
avoidance_balance =
    max(0, negative_rate - positive_rate)
```

Negative evidence strength:

```text
1 - exp(-N / 5)
```

Final avoidance:

```text
avoidance_balance
×
negative_evidence_strength
```

A minimum of **2 negative books** is required before a community can appear in
the avoidance ranking.

This is an intentional conservative rule.

---

## Exploration strength

Exploration measures exposure without established preference.

Let:

```text
X = exposure-only books
A = preference evidence + exposure-only books
```

Then:

```text
exploration_rate = X / A
```

Exploration evidence strength:

```text
1 - exp(-X / 5)
```

Final exploration:

```text
exploration_rate
×
exploration_evidence_strength
```

This prevents raw TBR volume from being interpreted as preference.

It also captures the useful distinction between:

```text
strong established interest + continued exploration
```

and:

```text
primarily exploratory exposure
```

---

## Evidence strength

Evidence strength is direction-independent.

```text
actionable evidence =
    preference evidence
    + exposure-only evidence
```

Then:

```text
evidence_strength =
    1 - exp(-actionable evidence / 5)
```

This provides a confidence-like measure without turning confidence into a
preference score.

---

# Model comparison

Three approaches were considered during development.

### Model A — raw positive rate

```text
positive / community books
```

Rejected as the primary signal because one-book communities could dominate the
ranking.

### Model B — raw positive count

```text
positive book count
```

Rejected as the primary signal because it measures volume more than preference.

### Model C — evidence-adjusted multi-signal model

The current model separates:

- preference
- avoidance
- exploration
- evidence strength

This is the selected scoring contract for the Reading DNA experiment.

---

# Validation results

## Preference

Preference rankings became substantially more interpretable after evidence
smoothing.

For the current user:

```text
Community 0
    score: 0.9217
    positive: 43
    negative: 0
    preference evidence: 43

Community 11
    score: 0.6252
    positive: 10
    negative: 1
    preference evidence: 11

Community 8
    score: 0.5180
    positive: 7
    negative: 0
    preference evidence: 7
```

These rankings are consistent with the desired behavior: stronger evidence
produces stronger preference signals.

---

## Avoidance

No community qualified for avoidance for any of the three development
readers.

Validation:

```text
Avoidance validation: PASS
Zero-negative communities with positive avoidance score: 0
Zero-negative communities ranked for avoidance: 0
```

This is not considered a model failure.

The conservative interpretation is that the current three-reader validation
set does not contain enough community-level negative evidence to claim
meaningful avoidance.

The model should not manufacture avoidance merely to populate a dashboard
section.

---

## Exploration

The exploration signal behaves differently from preference.

For the current user:

```text
Community 28
    exploration: 0.9257
    preference evidence: 0
    exposure-only: 13

Community 9
    exploration: 0.7761
    preference evidence: 6
    exposure-only: 22

Community 0
    exploration: 0.6767
    preference evidence: 43
    exposure-only: 90
```

The Community 0 result is especially useful conceptually.

The reader has strong established preference evidence there while also having a
large number of exposure-only books. The model therefore identifies:

> established interest + continued exploration

rather than treating all exposure as evidence of unknown preference.

---

# Decision

**Freeze the scoring model.**

The following are now part of the Reading DNA experiment contract:

```text
Preference:
    smoothed positive preference
    ×
    preference evidence strength

Avoidance:
    max(0, negative rate - positive rate)
    ×
    negative evidence strength
    with minimum 2 negative books

Exploration:
    exposure-only proportion
    ×
    exploration evidence strength

Evidence:
    amount of actionable evidence
```

Do not combine these signals into a single Reading DNA score.

---

# Known limitations

### Three-reader validation set

The current validation population is too small and too controlled to establish
generalization.

A later validation phase should include more diverse reader profiles.

### Community IDs are not semantic labels

Community IDs are arbitrary and can change when the graph configuration
changes.

The application should eventually attach interpretable descriptions to
communities using their member books and attributes.

### Avoidance is intentionally sparse

The current sample produces no qualifying avoidance communities.

That is preferable to making unsupported claims, but it should be revisited
with a larger reader population.

### Exploration is not preference

A large TBR or current-reading population does not mean a reader likes the
community.

Exposure remains a separate evidence stream.

---

# Next step

Build the actual reader-level Reading DNA representation from these signals.

The next layer should answer:

```text
What are this reader's established interests?

What are their strongest emerging/exploratory areas?

Which areas have enough negative evidence to mention avoidance?

How strong is the evidence behind each statement?

Which overlapping attributes help explain those interests?
```

Only after that representation is stable should it become the primary input to
the public-facing dashboard and recommendation explanations.
