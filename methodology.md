# Methodology

## 1. Business Problem

The project framed customer account closure as a binary classification problem.

The target variable was:

- **Active** — account remained active
- **Closed** — account had closed

The analytical objective was to compare classification models and determine which model better identified customers in the Closed class.

## 2. Data Preparation

The original dataset contained **10,119 customer records**.

Class distribution:

| Status | Records | Share |
| --- | ---: | ---: |
| Active | 8,495 | 83.95% |
| Closed | 1,624 | 16.05% |

The identifier field `rowID` was removed before modelling.

The categorical fields explicitly retained in the project workflow were:

- `gender`
- `marital_status`
- `account_type`
- `annual_income`

They were converted to dummy variables using `pd.get_dummies(..., drop_first=True)`.

Other numeric predictors were retained.

## 3. Train-Test Split

The encoded feature matrix and target were split using:

```python
train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produced a test set of **2,024 observations**:

- Active: 1,699
- Closed: 325

Stratification preserved the original class proportions across training and testing data.

## 4. Logistic Regression

Numeric features were standardised using `StandardScaler`.

The retained Logistic Regression configuration was:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

Balanced class weights were used because Closed customers represented a minority of the dataset.

### Logistic Regression Test Results

Confusion matrix:

```text
[[1251, 448],
 [  85, 240]]
```

Selected metrics:

- Accuracy: 73.67%
- Balanced accuracy: 73.74%
- Active precision: 93.64%
- Active recall: 73.63%
- Active F1: 82.44%
- Closed precision: 34.88%
- Closed recall: 73.85%
- Closed F1: 47.38%

## 5. Decision Tree

A Decision Tree classifier was trained and evaluated using the same held-out test data.

The original retained outputs contain the model's evaluation results, but the exact constructor parameters were not preserved in the portfolio source material. They are therefore not reconstructed.

### Decision Tree Test Results

Confusion matrix:

```text
[[1526, 173],
 [  28, 297]]
```

Selected metrics:

- Accuracy: 90.07%
- Balanced accuracy: 90.60%
- Active precision: 98.20%
- Active recall: 89.82%
- Active F1: 93.82%
- Closed precision: 63.19%
- Closed recall: 91.38%
- Closed F1: 74.72%

## 6. Evaluation

Because the target was imbalanced, accuracy alone was not sufficient for comparing models.

The comparison focused on:

- accuracy
- balanced accuracy
- confusion matrix
- class-specific precision
- class-specific recall
- class-specific F1 score

The Decision Tree was stronger across the retained test-set metrics and substantially reduced both false positives and false negatives.

## 7. Limitation

These results describe performance on the retained 20% test split used in the project. A production implementation would require additional validation on new data and ongoing monitoring for model drift and changes in customer behaviour.
