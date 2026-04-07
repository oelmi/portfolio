# Article Linking Error Analysis

## Overview

This project focuses on the evaluation of a machine learning system for linking news articles to official statistical publications. The broader system combines lexical and semantic matching methods to rank candidate matches, while this repository highlights my individual contribution: error analysis and model auditing.

The full group project achieved a strong improvement over the original baseline, increasing AUC from 0.484 to 0.908 and reaching a Success@5 score of 0.927. My work focused on understanding where and why the system still fails, especially in high-confidence error cases.

## My Contribution

My main contribution was building and analyzing the error analysis pipeline.

I worked on:
- identifying high-confidence false positives
- auditing model predictions at topic level
- mapping outputs to a taxonomy for interpretable error inspection
- supporting analysis of label noise and systematic mismatch patterns

This contribution helped move the project beyond model performance alone and toward understanding dataset quality, failure modes, and practical reliability.

## Methods

The broader project used a hybrid matching setup with lexical and semantic signals. My analysis pipeline focuses on post-model evaluation.

Key elements used in my contribution:
- group-aware train/test splitting
- feature scaling with `MinMaxScaler`
- `CatBoostClassifier` for prediction scoring
- topic resolution using taxonomy matching
- export of structured error outputs for manual inspection

The error analysis script inspects model behavior by comparing assigned topics with inferred ground-truth topics and surfacing representative correct and incorrect cases.

## Results

The full project showed that the best hybrid setup substantially outperformed the original baseline:
- Baseline AUC: 0.484
- Best AUC: 0.908
- Success@5: 0.927

The error analysis contributed to interpreting these results by showing that some high-confidence mistakes were not purely model failures, but were linked to:
- inconsistent labels in the dataset
- semantically similar articles assigned to different topics
- boilerplate or repeated institutional phrasing that inflated similarity

This made the analysis useful not only for model evaluation, but also for improving trust in the system and identifying data quality issues.

## Repository Structure

```text
.
├── README.md
├── src/
│   └── error_analysis.py
├── results/
│   └── sample_outputs/
└── assets/
    └── figures/

## Notes on Collaboration and Privacy

This project was originally developed as part of a university group project.

This repository contains only my individual contributions, specifically focused on error analysis and model evaluation.

To respect collaboration boundaries and data privacy, the following are not included:
- the full group report
- contributions from other team members
- private or restricted datasets
- internal project materials not owned by me

All included code and materials are either my own work or have been cleaned and anonymized for demonstration purposes.
