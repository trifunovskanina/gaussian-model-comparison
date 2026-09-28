# A Comparative Evaluation of Gaussian Naive Bayes, Linear Discriminant Analysis and Quadratic Discriminant Analysis for Bank Term Deposit Prediction Under Class Imbalance

A statistical machine learning study investigating how different **Gaussian assumptions**, **feature independence** and **covariance structure** affect classification performance.

---

## Technologies

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## Overview

This project compares three probabilistic classifiers:

* **Gaussian Naive Bayes**
* **Linear Discriminant Analysis**
* **Quadratic Discriminant Analysis**

Rather than focusing only on accuracy, the project examines how the models' statistical assumptions influence their predictive behavior, especially when the target is **imbalanced**.

---

<h2 align="center">Mathematical Foundations</h2>

<h3 align="center">Gaussian Naive Bayes</h3>

```math
P(y \mid X) =
\frac{P(y)\prod_{i=1}^{n}P(x_i\mid y)}
{P(X)}
```

<h3 align="center">Linear Discriminant Analysis</h3>

```math
\Sigma_1 = \Sigma_2 = \cdots = \Sigma_k = \Sigma
```

<h3 align="center">Quadratic Discriminant Analysis</h3>

```math
\Sigma_1, \Sigma_2, \ldots, \Sigma_k
```

---

### Research Question

**How do the statistical assumptions of Gaussian Naive Bayes, Linear Discriminant Analysis and Quadratic Discriminant Analysis affect their predictive performance under class imbalance?**

### Hypothesis

**Because all three models make different assumptions about feature distributions, feature independence and covariance structure, their predictive performance will differ.**

---

## About the Dataset

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to assess if the product would be subscribed or not.

The target variable is:
- `y = yes` - client subscribed to a term deposit
- `y = no` - client did not subscribe to a term deposit

The target is highly imbalanced, with about **88.7%** negative cases, and **11.3%** positive cases.

---

## Methodology

1. **Data Preprocessing**
2. **Exploratory Data Analysis**
3. **Multicollinearity Analysis**
4. **Feature Selection**
5. **Data Leakage Prevention**
6. **Train-Test Split**
7. **Feature Normalization**
8. **Model Evaluation**
9. **Cross-Validation**
10. **Class Balancing**
11. **Comparative Analysis**

---

## Models

| Model                               | Key Assumption                                                                         |
|-------------------------------------|----------------------------------------------------------------------------------------|
| **Gaussian Naive Bayes**            | Features are **conditionally independent** and individually Gaussian within each class |
| **Linear Discriminant Analysis**    | Class distributions are Gaussian with a **shared covariance matrix**                   |
| **Quadratic Discriminant Analysis** | Class distributions are Gaussian with a **separate covariance matrix** for each class  |

The models provide a useful comparison because they share the Gaussian assumption, while differing in how they assume relationships between features.

---

## Evaluation Metrics

The models are evaluated using **accuracy** and **minority-class recall**.

### Accuracy

Accuracy measures the proportion of all predictions that are classified correctly:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Minority-Class Recall

Recall measures the proportion of actual minority-class observations that are correctly identified:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Because the positive class represents clients who subscribed to a term deposit and is the minority class, minority-class recall is particularly important for evaluating how effectively the models identify potential subscribers.

---

## Multicollinearity Analysis

Variance Inflation Factor (VIF) was used to assess **linear multicollinearity** among the numerical features.

The features with the highest VIF values were:

| Feature | VIF |
|---|---:|
| `nr.employed` | 26746.63 |
| `cons.price.idx` | 22561.12 |
| `euribor3m` | 226.23 |

These three features were removed to reduce the strongest multicollinearity among the predictors.

After removing them, the VIF values of the remaining features decreased substantially.

---

## Data Leakage

The `duration` feature is removed because it represents the length of the phone call.

This information is only available after the call ends and therefore would not be known when deciding whether to contact a client. 

---

## Results

The models were evaluated using the original training data, then again after undersampling and SMOTE. Accuracy and minority-class (`1`) recall are emphasized because accuracy alone can be misleading when the target variable is imbalanced.

### Before Balancing

| Model                               | Accuracy | Minority-Class Recall | Minority-Class F1 |
| ----------------------------------- | -------: | --------------------: | ----------------: |
| **Gaussian Naive Bayes**            |     0.88 |                  0.27 |              0.35 |
| **Linear Discriminant Analysis**    | **0.90** |                  0.23 |              0.33 |
| **Quadratic Discriminant Analysis** |     0.88 |              **0.33** |          **0.39**|

### After Undersampling

| Model                               | Accuracy | Minority-Class Recall | Minority-Class F1 |
| ----------------------------------- | -------: | --------------------: | ----------------: |
| **Gaussian Naive Bayes**            |     0.84 |                  0.38 |              0.35 |
| **Linear Discriminant Analysis**    |     0.74 |              **0.72** |              0.39 |
| **Quadratic Discriminant Analysis** | **0.88** |                  0.37 |          **0.41** |

### After Oversampling (SMOTE)

| Model | Accuracy | Minority-Class Recall | Minority-Class F1 |
| ----------------------------------- | -------: | --------------------: | ----------------: |
| **Gaussian Naive Bayes** | 0.83 | 0.40 | 0.35 |
| **Linear Discriminant Analysis** | 0.74 | **0.63** | 0.36 |
| **Quadratic Discriminant Analysis** | **0.88** | 0.41 | **0.43** |

---

## Cross-Validation

Five-fold **stratified cross-validation** was performed on the training data to assess the consistency of model performance across different data splits. Stratification preserves the class distribution across folds. Model performance is reported as the mean and standard deviation across the five folds.

| Model | Accuracy | Minority-Class Recall | Minority-Class F1 |
| ----------------------------------- | -------: | --------------------: | ----------------: |
| **Gaussian Naive Bayes**            | 0.88 ± 0.00 | 0.28 ± 0.01 | 0.35 ± 0.01 |
| **Linear Discriminant Analysis**    | 0.89 ± 0.00 | 0.23 ± 0.01 | 0.33 ± 0.01 |
| **Quadratic Discriminant Analysis** | 0.88 ± 0.00 | 0.33 ± 0.01 | 0.38 ± 0.01 |

---

## Key Findings

The imbalanced dataset produces **high overall accuracy** while minority-class **recall remains low**. 

After undersampling, minority-class recall increases for all three models, while overall accuracy differs between models. **Linear Discriminant Analysis** had the most substantial change, with an increase in minority-class recall from **0.23** to **0.72**, while 
accuracy decreased from **0.90** to **0.74**. **Quadratic Discriminant Analysis** showed the smallest changes, accuracy remaining **0.88**, with minority-class recall increase from **0.33** to **0.37**. **Gaussian Naive Bayes** experienced a decrease in accuracy from **0.88** to **0.84**, and a jump in minority-class recall from **0.27** to **0.38**.

After **Synthetic Minority Oversampling Technique** (SMOTE), **Linear Discriminant Analysis**' accuracy decreased from **0.90** to **0.74** with minority-class recall increasing from **0.23** to **0.63**. **Quadratic Discriminant Analysis**' accuracy remained the most stable at **0.88**, while minority-class recall increased from **0.33** to **0.41**. **Gaussian Naive Bayes** had an accuracy decrease from **0.88** to **0.83**, with an increase in minority-class recall from **0.27** to **0.40**

In both techniques, **Quadratic Discriminant Analysis** performed with the highest accuracy and minority-class F1-score, while **Linear Discriminant Analysis** achieved the highest minority-class recall.

---

## One-Hot Encoded Features

A limitation is that the one-hot encoded categorical variables are binary and therefore do not follow a Gaussian distribution per the assumptions of the models. Removing the one-hot encoded categorical features **improved** accuracy, minority-class recall and F1-score for all three models.

| Model | Original Accuracy | Without One-Hot Accuracy | Original Minority Recall | Without One-Hot Recall | Original Minority F1 | Without One-Hot F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Gaussian Naive Bayes | 0.88 | **0.90** | 0.27 | **0.45** | 0.35 | **0.49** |
| Linear Discriminant Analysis | 0.90 | **0.91** | 0.23 | **0.43** | 0.33 | **0.51** |
| Quadratic Discriminant Analysis | 0.88 | **0.89** | 0.33 | **0.54** | 0.38 | **0.52** |
