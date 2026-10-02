# AeroBank Customer Churn Prediction

![AeroBank Customer Churn Prediction](aerobank-churn-preview.png)

**Python · Jupyter · Machine Learning · Classification**

## Project Overview

This project predicts whether an AeroBank customer account will remain **Active** or become **Closed** using supervised machine learning.

The analysis compares two classification models:

- Logistic Regression
- Decision Tree

The business objective is to identify customers at higher risk of account closure so retention activity can be focused earlier and more efficiently.

## Dataset

The dataset contained **10,119 customer records**:

- **8,495 Active customers** — 83.95%
- **1,624 Closed customers** — 16.05%

The target variable was `status`, with the two classes **Active** and **Closed**.

During preprocessing:

- `rowID` was removed
- categorical variables including `gender`, `marital_status`, `account_type`, and `annual_income` were encoded
- numeric variables were retained for modelling
- the data was split into **80% training / 20% testing**
- the split used **stratification** to preserve the class distribution
- `random_state=42` was used for reproducibility

The final test set contained **2,024 customers**:

- 1,699 Active
- 325 Closed

## Modelling Approach

### Logistic Regression

The Logistic Regression pipeline used:

- one-hot encoding with `drop_first=True`
- `StandardScaler`
- `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`

Class weighting was used to reduce the effect of the imbalance between Active and Closed customers.

### Decision Tree

A Decision Tree classifier was trained as the second model and evaluated on the same stratified test set.

The original notebook confirms the model configuration:

```python
DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
```

The depth limit was used to constrain model complexity, while balanced class weights accounted for the smaller Closed-customer class.

## Model Comparison

| Metric | Logistic Regression | Decision Tree |
| --- | ---: | ---: |
| Accuracy | 73.67% | **90.07%** |
| Balanced Accuracy | 73.74% | **90.60%** |
| Closed Precision | 34.88% | **63.19%** |
| Closed Recall | 73.85% | **91.38%** |
| Closed F1 | 47.38% | **74.72%** |
| False Positives | 448 | **173** |
| False Negatives | 85 | **28** |

The Decision Tree produced the stronger overall result on the held-out test set.

## Confusion Matrices

### Logistic Regression

| Actual \ Predicted | Active | Closed |
| --- | ---: | ---: |
| Active | 1,251 | 448 |
| Closed | 85 | 240 |

The Logistic Regression model correctly identified **240 of 325 Closed customers**, but it also classified **448 Active customers as Closed**.

### Decision Tree

| Actual \ Predicted | Active | Closed |
| --- | ---: | ---: |
| Active | 1,526 | 173 |
| Closed | 28 | 297 |

The Decision Tree correctly identified **297 of 325 Closed customers** while reducing false positives to **173**.

## Key Findings

The Decision Tree improved test accuracy by approximately **16.4 percentage points** compared with Logistic Regression.

It also:

- increased recall for Closed customers from approximately **74% to 91%**
- reduced missed Closed customers from **85 to 28**
- reduced false closure alerts from **448 to 173**
- increased the Closed-class F1 score from approximately **0.47 to 0.75**

For this dataset and test split, the Decision Tree provided the stronger balance between identifying likely account closures and limiting incorrect churn alerts.

## Business Interpretation

For a customer-retention use case, recall for the **Closed** class is important because a false negative represents a customer who closes their account without being identified by the model.

The Decision Tree substantially reduced those missed closures while also producing fewer false positives than Logistic Regression. This makes it the stronger model of the two for prioritising customers for further retention review in this case study.

## Repository Structure

```text
aerobank-customer-churn-prediction/
├── README.md
├── aerobank-churn-preview.png
├── aerobank_churn_analysis.ipynb
├── methodology.md
├── model_comparison.csv
└── confusion_matrices.csv
```

## Skills Demonstrated

**Python · Jupyter · pandas · scikit-learn · Data Preprocessing · One-Hot Encoding · Feature Scaling · Logistic Regression · Decision Trees · Classification · Class Imbalance · Confusion Matrices · Precision · Recall · F1 Score · Model Evaluation · Business Analytics**

## Project Context

This project was completed in **2026** as part of **MIS140** at Deakin University and has been reformatted as a professional portfolio case study.

Personal identifiers and university submission material are excluded from this public repository.
