
# Breast Cancer Detection

**Author:** Guohao (Rious) Yang  
**Department:** Computer Science, Kean University  
**Course:** CPS 4882  
**Instructor:** Dr. Amani Ayad  
**Date:** May 2, 2025

---

## Project Overview

This project focuses on early detection of breast cancer by applying machine learning and deep learning models to the UCI Wisconsin Breast Cancer dataset.  
The goal is to build highly accurate classifiers to improve patient outcomes through early diagnosis.

---

## Dataset

- Source: UCI Breast Cancer Wisconsin Diagnostic Dataset
- 569 instances, 32 features (radius, texture, area, smoothness, etc.)
- Target variable: Diagnosis (M for Malignant, B for Benign)

---

## Data Preprocessing

- Removed the ID column (not predictive)
- Encoded Diagnosis: M=1, B=0
- Standardized all feature values
- Removed missing values, duplicates, and outliers
- Balanced classes using SMOTE (Synthetic Minority Oversampling Technique)
- Selected features based on correlation, feature importance, and linearity analysis (R² and p-value)

---

## Machine Learning Models

Implemented and evaluated the following models:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC

**Observations:**
- Logistic Regression achieved the highest baseline performance.
- XGBoost and Random Forest showed the best AUC-ROC scores.
- Recall was emphasized to minimize false negatives (missing malignant tumors).

---

## Deep Learning Models

Two deep learning models were built:

| Framework | Details |
|-----------|---------|
| TensorFlow | Built using Keras Sequential API |
| PyTorch | Custom model using nn.Module |

**Common Model Architecture:**

- Input layer (matching feature count)
- Dense(64) + ReLU
- Dropout(30%)
- Dense(32) + ReLU
- Output Dense(1) + Sigmoid

**Evaluation metrics:** Accuracy, AUC-ROC, Precision, Recall, F1-score

**Results:**  
Deep learning models achieved over 97% accuracy. Feature selection further improved generalization.

---

## Results Summary

| Model                | Data Type         | Accuracy | Recall | F1-score | AUC-ROC |
|----------------------|-------------------|----------|--------|----------|---------|
| Logistic Regression  | Raw Data           | 96.49%   | 95.35% | 95.92%   | 0.985   |
| SVM                  | Raw Data           | 96.49%   | 95.35% | 95.92%   | 0.984   |
| Random Forest        | Raw Data           | 96.49%   | 95.35% | 95.92%   | 0.987   |
| XGBoost              | Raw Data           | 96.49%   | 95.35% | 95.92%   | 0.989   |
| TensorFlow DNN       | Raw Data           | 97.37%   | 97.67% | 97.62%   | 0.991   |
| PyTorch DNN          | Raw Data           | 97.37%   | 97.67% | 97.62%   | 0.990   |
| TensorFlow DNN       | Processed Features | 98.25%   | 97.67% | 97.83%   | 0.993   |
| PyTorch DNN          | Processed Features | 98.25%   | 97.67% | 97.83%   | 0.993   |

---

## Key Techniques

- SMOTE oversampling
- Feature selection using R² and p-values
- StandardScaler normalization
- Deep learning with TensorFlow and PyTorch
- Dropout regularization

---

## Conclusion

- Logistic Regression achieved excellent baseline performance.
- XGBoost and Random Forest provided strong AUC-ROC scores.
- TensorFlow and PyTorch deep learning models outperformed traditional models with over 98% accuracy after feature selection.
- Data balancing, feature engineering, and regularization were crucial to the project's success.

---
