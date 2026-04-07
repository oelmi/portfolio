# Thesis Design – Implicit Reference Linking

## Problem

Parliamentary debates frequently contain references to other governmental documents such as reports, motions, and legislative proposals. While explicit references can be detected using rule-based systems, many references are implicit and expressed through descriptive language.

For example:
- “the report discussed last week”
- “the evaluation sent earlier this year”

These implicit references are difficult to detect because they require contextual and semantic understanding rather than simple pattern matching :contentReference[oaicite:0]{index=0}.

---

## Objective

The goal of this thesis is to improve the detection and linking of implicit references in Dutch parliamentary proceedings using retrieval-based machine learning methods.

The research investigates whether modern information retrieval and neural models can outperform existing rule-based and heuristic approaches.

---

## Approach

The system is designed as a **two-stage retrieval pipeline**:

### 1. Candidate Generation
Generate a set of potential target documents for each reference using:
- BM25 (sparse retrieval)
- Dense retrieval (e.g. SBERT / DPR)
- Hybrid retrieval (combining sparse and dense methods)

### 2. Reranking
Refine the candidate list using:
- Cross-encoder models (BERT-based reranking)

This architecture allows efficient retrieval followed by more precise ranking.

---

## Data

The project uses:
- Dutch parliamentary proceedings (Handelingen)
- Official government documents (e.g. motions, reports)
- A gold-standard dataset of verified reference links

---

## Evaluation

The system is evaluated using standard information retrieval metrics:

### Candidate Generation
- Recall@20
- Recall@100

### Full Pipeline
- Precision
- Recall
- F1-score
- Mean Reciprocal Rank (MRR)
- Hits@k

---

## Research Questions

The thesis investigates:

- How do sparse, dense, and hybrid retrieval methods compare in recall?
- How much does neural reranking improve performance?
- Does structured metadata improve retrieval quality?
- What types of implicit references remain difficult to detect?

---

## Key Design Choices

- Treat implicit reference detection as a **retrieval problem**
- Use a **two-stage architecture** (efficient + accurate)
- Combine **textual similarity with structured metadata**
- Include **error analysis** to understand failure cases

---

## Expected Contribution

This work aims to:

- Improve linking performance for implicit references  
- Provide a systematic comparison of retrieval strategies  
- Show how metadata can enhance retrieval pipelines  
- Identify common failure cases in implicit reference detection  

---

## Key Insight

Implicit reference detection is fundamentally a **semantic retrieval problem under ambiguity**.

Rule-based systems fail when references are descriptive or paraphrased, while retrieval-based neural models offer a more flexible and scalable solution.

---

## Notes

This is an active research project. The implementation and experiments are developed iteratively and updated throughout the thesis.
