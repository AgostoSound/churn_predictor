# Customer Churn Predictor

This project predicts the probability that a customer will churn (i.e., stop using a service) and estimates the potential revenue loss associated with that churn. The goal is to build a machine learning model from scratch using open data, and simulate business impact based on predicted outcomes.

##  Problem Statement

Customer retention is a key challenge in subscription-based and service-driven businesses. Predicting churn enables companies to take proactive measures to reduce losses and increase customer satisfaction.

## Tech Stack

- **Python** (v3.11)
- **pandas**, **numpy** – data manipulation
- **scikit-learn** – machine learning models
- **keras** + **tensorflow** – deep learning models
- **matplotlib**, **seaborn** – data visualization
- **Jupyter** – EDA and model development

## Features

- Binary classification: churned / not churned
- Business KPI simulation: revenue loss vs retention effort
- Class imbalance handling (SMOTE)
- Feature importance & explainability (SHAP or LIME)
- Interactive dashboard with Streamlit

##  Dataset

It uses the **Telco Customer Churn** dataset (publicly available on Kaggle).

## Goal

Build an end-to-end churn prediction pipeline:
- Perform EDA
- Train and evaluate models
- Interpret results
- Simulate business impact
