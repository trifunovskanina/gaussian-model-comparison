# A Comparative Evaluation of Gaussian Naive Bayes, Linear Discriminant Analysis and Quadratic Discriminant Analysis for Bank Term Deposit Prediction Under Class Imbalance

A statistical machine learning study investigating how different **Gaussian assumptions**, **feature independence** and **covariance assumptions** affect classification performance.

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

The models are evaluated both before and after **undersampling** the majority class.

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
\Sigma_1 \neq \Sigma_2 \neq \cdots \neq \Sigma_k
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
3. **Feature Selection**
4. **Train-Test Split**
5. **Feature Normalization**
6. **Model Evaluation**
7. **Class Balancing**

---

## Models

| Model                               | Key Assumption                                                                         |
|-------------------------------------|----------------------------------------------------------------------------------------|
| **Gaussian Naive Bayes**            | Features are **conditionally independent** and individually Gaussian within each class |
| **Linear Discriminant Analysis**    | Class distributions are Gaussian with a **shared covariance matrix**                   |
| **Quadratic Discriminant Analysis** | Class distributions are Gaussian with a **separate covariance matrix** for each class  |

The models provide a useful comparison because they share the Gaussian assumption, while differing in how they assume relationships between features.

---

## Data Leakage

The `duration` feature is removed because it represents the length of the phone call.

This information is only available after the call ends and therefore would not be known when deciding whether to contact a client. 

---

## Results

The models were evaluated using the original training data and after undersampling the majority class. Accuracy and minority-class (`1`) recall are emphasized because accuracy alone can be misleading when the target variable is imbalanced.

### Before Undersampling

| Model                               | Accuracy | Minority-Class Recall | Minority-Class F1 |
| ----------------------------------- | -------: | --------------------: | ----------------: |
| **Gaussian Naive Bayes**            |     0.88 |                  0.31 |              0.37 |
| **Linear Discriminant Analysis**    | **0.89** |                  0.33 |              0.41 |
| **Quadratic Discriminant Analysis** |     0.88 |              **0.39** |          **0.43** |

### After Undersampling

| Model                               | Accuracy | Minority-Class Recall | Minority-Class F1 |
| ----------------------------------- | -------: | --------------------: | ----------------: |
| **Gaussian Naive Bayes**            |     0.84 |                  0.45 |              0.39 |
| **Linear Discriminant Analysis**    |     0.77 |              **0.70** |              0.41 |
| **Quadratic Discriminant Analysis** | **0.88** |                  0.44 |          **0.45** |

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

## Key Findings

The imbalanced dataset produces **high overall accuracy** while minority-class **recall remains low**. 
After undersampling, minority-class recall increases for all three models, while overall accuracy differs between models.

**Linear Discriminant Analysis** had the most substantial change, with an increase in minority-class recall from **0.33** to **0.70**, while 
accuracy decreased from **0.89** to **0.77**. 

**Quadratic Discriminant Analysis** had the most stable results,
 accuracy remaining **0.88**, with minority-class recall increase from **0.39** to **0.44**.

**Gaussian Naive Bayes** experienced a decrease in accuracy from **0.88** to **0.84** and a jump in minority-class recall from **0.31** to **0.45**.

---

**Note**: A limitation is that the one-hot encoded categorical variables are binary and therefore do not follow a Gaussian distribution per the assumptions of the models.
