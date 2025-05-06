# 3y_project
This repository contains Python scripts used to evaluate various classification models on the UCI Heart Disease dataset. The models are tested both **with** and **without** SMOTE (Synthetic Minority Over-sampling Technique) to compare performance impacts.

## File Naming Convention

Each script follows the structure:

- `w_smote`: SMOTE is applied
- `wo_smote`: SMOTE is not applied

## Overview of Scripts

| Filename                          | Description |
|----------------------------------|-------------|
| `frb_w_smote.py`                 | Fuzzy-Ranked Based ensemble model with SMOTE |
| `frb_wo_smote.py`                | Fuzzy-Ranked Based ensemble model without SMOTE |
| `moe_w_smote.py`                 | Mixture of Experts ensemble model with SMOTE |
| `moe_wo_smote.py`                | Mixture of Experts ensemble model without SMOTE |
| `bagging_lr_svm_nb_w_smote.py`  | Bagging ensemble model with SMOTE |
| `bagging_lr_svm_nb_wo_smote.py` | Bagging ensemble model without SMOTE |
| `svm_features_w_smote.py`       | SVM using feature selection with SMOTE |
| `svm_features_wo_smote.py`      | SVM using feature selection without SMOTE |
| `lr_features_w_smote.py`        | Logistic Regression using feature selection with SMOTE |
| `lr_features_wo_smote.py`       | Logistic Regression using feature selection without SMOTE |
| `nb_features_w_smote.py`        | Naive Bayes using feature selection with SMOTE |
| `nb_features_wo_smote.py`       | Naive Bayes using feature selection without SMOTE |

## Data

The dataset used for all evaluations is:

```
datafile/heart_disease_uci.csv
```

## Requirements

Install the required Python packages before running the scripts:

```bash
pip install scikit-learn imbalanced-learn numpy pandas matplotlib seaborn tensorflow keras