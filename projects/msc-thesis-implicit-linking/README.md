# Implicit Reference Linking (MSc Thesis)

## Overview

This project focuses on detecting and linking implicit references in Dutch parliamentary documents.

Unlike explicit references (e.g. document IDs), implicit references are expressed through descriptive language such as “the report discussed last week”. These references require semantic understanding and contextual reasoning, making them challenging to detect using rule-based approaches.

The goal of this thesis is to improve reference linking using retrieval-based machine learning methods.

---

## My Contribution

I am developing a retrieval-based pipeline for detecting and linking implicit references.

Key components of my work include:
- designing a two-stage retrieval system  
- implementing and comparing sparse, dense, and hybrid retrieval methods  
- applying neural reranking models  
- evaluating system performance using standard IR metrics  
- performing error analysis to understand failure cases  

---

## Methods

The system follows a two-stage retrieval architecture:

### 1. Candidate Generation
- BM25 (sparse retrieval)  
- Dense retrieval (e.g. Sentence-BERT / DPR)  
- Hybrid retrieval methods  

### 2. Reranking
- Cross-encoder models for fine-grained ranking  

### Data
- Dutch parliamentary proceedings (Handelingen)  
- Official government documents  
- Gold-standard dataset of verified reference links  

### Evaluation Metrics
- Recall@k  
- Mean Reciprocal Rank (MRR)  
- F1-score  

---

## Thesis Design

A structured overview of the research design and methodology is available here:

`reports/thesis_design.md`

---

## Current Progress

- dataset exploration and preparation  
- baseline retrieval models (BM25)  
- initial experiments with dense retrieval  
- evaluation framework setup  

Planned next steps include improving retrieval quality, integrating metadata, and optimizing reranking models.

---

## Key Insight

Implicit reference detection can be formulated as a retrieval problem under semantic ambiguity.

Traditional rule-based systems struggle with descriptive and paraphrased references, while retrieval-based neural models provide a more flexible and scalable approach.

---

## Repository Structure

```text
.
├── README.md
├── notebooks/        # experiments and analysis (including EDA)
├── reports/          # thesis design and documentation
├── src/              # (planned) retrieval and modeling code


```
## Notes
This repository is an active research project and is continuously updated as the thesis progresses.

Sensitive data and internal materials are not included.

## Tech Stack

- Python  
- PyTorch  
- BM25  
- Information Retrieval techniques  
