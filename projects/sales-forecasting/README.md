# Sales Forecasting (Time Series)

## Overview

This project focuses on forecasting daily retail sales using a subset of Walmart’s M5 dataset. The goal is to predict future demand while accounting for seasonality, sparsity, and external factors such as pricing and events.

The project compares classical statistical models, machine learning approaches, and deep learning methods to evaluate their effectiveness on real-world time series data.

---

## My Contribution

My contribution begins from the exploratory data analysis stage and continues through model selection, forecasting, and evaluation.

I worked on:
- exploratory data analysis (EDA)
- identifying important demand patterns such as seasonality and sparsity
- selecting suitable forecasting models
- generating forecasts
- comparing model performance across approaches

This part of the project focuses on understanding the structure of the data and choosing models that fit the forecasting problem effectively.

---

## Methods

The project evaluates multiple forecasting approaches:

### Classical Models
- Seasonal Naive
- Exponential Smoothing (ETS)

### Machine Learning Models
- XGBoost
- LightGBM

### Deep Learning
- LSTM

Key inputs and features used in the forecasting setup include:
- lagged sales values
- rolling statistics
- calendar-based features
- price information
- event indicators

The main evaluation metric used is:

- Root Mean Squared Error (RMSE)

---

## Results

The best-performing model was **XGBoost**, achieving an RMSE of approximately **2.90** on the validation set.

Key findings:
- machine learning models outperformed classical statistical baselines
- deep learning did not perform best in this setting
- tree-based models handled sparsity and seasonality more effectively

---

## Key Insight

This project shows that model performance depends strongly on the structure of the data.

In this case, strong weekly seasonality and high sparsity made feature-based machine learning models more effective than more complex deep learning approaches. This highlights the importance of choosing models based on data characteristics rather than complexity alone.

---

## Repository Structure

```text
.
├── README.md
└── notebooks/
    └── forecasting_analysis.ipynb
```
## Notes on Collaboration and Privacy

This project was originally developed as part of a university group project.

This repository contains only my individual contribution, starting from the exploratory data analysis section through the end of the project.

To respect collaboration boundaries and privacy, the following are not included:

the full original group report
work completed by other team members before my contribution section
private or restricted datasets

All included materials are limited to my own work or cleaned versions prepared for portfolio purposes.

## Tech Stack
R
R Markdown
XGBoost
LightGBM
LSTM
Time series forecasting methods
