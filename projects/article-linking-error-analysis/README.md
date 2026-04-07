# Article Linking Error Analysis

## Overview

This project focuses on evaluating a machine learning system for linking news articles to official statistical publications. The broader system combines lexical and semantic matching methods to rank candidate matches, while this repository highlights my individual contribution: error analysis and model auditing.

The full project significantly improved performance over the original baseline, increasing AUC from 0.484 to 0.908 and achieving a Success@5 score of 0.927. My work focuses on understanding where and why the system still fails, particularly in high-confidence error cases.

---

## My Contribution

I designed and implemented an error analysis pipeline to evaluate model predictions and identify systematic failure patterns.

Key contributions:
- Identified high-confidence false positives
- Audited model predictions at topic level
- Mapped outputs to a taxonomy for interpretable error inspection
- Analyzed label noise and dataset inconsistencies

This work shifts the focus from model performance alone to understanding reliability, failure modes, and data quality.

---

## Methods

The broader system uses a hybrid matching approach combining lexical and semantic signals. My contribution focuses on post-model evaluation.

Key components:
- Group-aware train/test splitting
- Feature scaling using `MinMaxScaler`
- Prediction scoring using `CatBoostClassifier`
- Topic resolution via taxonomy mapping
- Structured export of error cases for analysis

The pipeline evaluates model behavior by comparing predicted and inferred ground-truth topics and surfacing representative correct and incorrect cases.

---

## Results

The hybrid system achieved strong improvements:
- Baseline AUC: 0.484  
- Best AUC: 0.908  
- Success@5: 0.927  

The error analysis revealed that many high-confidence errors were not purely model failures, but were caused by:

- Inconsistent labeling in the dataset  
- Semantically similar articles assigned to different topics  
- Repeated institutional phring ("boilerplate bias") inflating similarity  

---

## Key Insight

A major finding from this analysis is that some model “errors” actually expose weaknesses in the dataset rather than the model itself.  

This highlights the importance of combining model evaluation with data auditing when building real-world machine learning systems.

---

## Repository Structure

```text
.
├── README.md
├── src/
│   └── error_analysis.py
```
## Notes on Collaboration and Privacy

This project was originally developed as part of a university group project.

This repository contains only my individual contributions, specifically focused on error analysis and model evaluation.

To respect collaboration boundaries and data privacy, the following are not included:
- the full group report
- contributions from other team members
- private or restricted datasets
- internal project materials not owned by me

All included code and materials are either my own work or have been cleaned and anonymized for demonstration purposes.

## Tech Stack
Python
Pandas
NumPy
scikit-learn
CatBoost
