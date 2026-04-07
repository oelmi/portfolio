# IMDB Machine Learning Pipeline

## Overview

This project focuses on building an end-to-end machine learning pipeline to predict movie ratings using structured and external data sources.

The pipeline integrates multiple datasets, performs data cleaning and feature engineering, and evaluates machine learning models to predict IMDb ratings.

---

## My Contribution

My primary contribution was designing and implementing the feature engineering pipeline.

Key contributions:
- Created meaningful features from raw movie metadata
- Engineered features such as:
  - director success rate
  - genre-based patterns
  - temporal features (e.g. release year)
- Improved model performance through feature design
- Supported model evaluation by providing structured input data

This work enabled the model to capture underlying patterns in movie success beyond raw input data.

---

## Methods

The pipeline consists of the following stages:

1. Data ingestion and integration  
2. Data cleaning and preprocessing  
3. Feature engineering (my focus)  
4. Model training and evaluation  

Models used in the project include:
- XGBoost  
- LightGBM  

Feature engineering included:
- aggregation of categorical variables  
- encoding of genres and metadata  
- transformation of temporal features  
- integration of external data sources  

---

## Results

The project achieved strong predictive performance, with the best model reaching approximately **94% validation accuracy**.

This demonstrates the importance of feature engineering in improving model performance on structured datasets.

---

## Key Insight

Feature engineering had a major impact on model performance.

Rather than relying solely on model complexity, carefully designed features allowed the model to capture meaningful patterns such as:
- director reputation
- genre trends
- temporal effects in movie releases  

---

## Repository Structure

```text
.
├── README.md
├── src/
│   └── feature_engineering.py
```
## Notes on Collaboration and Privacy

This project was developed as part of a university group project.

This repository contains only my individual contributions, specifically focused on feature engineering.

To respect collaboration boundaries and data privacy, the following are not included:

the full group repository
contributions from other team members
private datasets or external APIs

All included materials are either my own work or have been cleaned for demonstration purposes.

## Tech Stack
Python
Pandas
NumPy
XGBoost
LightGBM
