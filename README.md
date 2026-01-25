# 🩺 Breast Cancer Detection: Logistic Regression Baseline

Building a high-recall classification system to detect malignancy using the Breast Cancer Wisconsin dataset.

---

## 🎯 Objective

The goal of this project is to establish a rigorous baseline for medical diagnosis.

In clinical AI, the cost of a **False Negative** (missing a cancer case) is far more dangerous than a **False Positive**.

This project focuses on:

- Transitioning from regression into **Binary Classification**
- Handling **Class Imbalance** in real-world medical datasets
- Prioritizing **Recall** over raw Accuracy

---

## 📊 Classification Task

The model predicts whether a tumor is:

- **0 → Malignant (Cancerous)**
- **1 → Benign (Healthy)**

Unlike regression (predicting a number), this model predicts a **category**.

---

## ## Dataset (Real Medical Data)

We use the **Breast Cancer Wisconsin (Diagnostic) dataset** from `sklearn`.

Dataset characteristics:

- **569 patients**
- **30 numeric tumor features**
- Features describe size, shape, texture, and irregularities
- Each patient has a known diagnosis (label)

This is a **real-world medical dataset**, not synthetic data.

---

## ⚙️ Project Pipeline

The workflow follows a standard medical ML baseline:

- **Data Loading**  
  Partitioning the dataset into Features ($X$) and Labels ($y$)

- **Class Balance Check**  
  Visualizing label distribution to detect imbalance bias

- **Feature Correlation**  
  Heatmap analysis to identify the strongest tumor predictors

- **Stratified Train/Test Split**  
  80/20 split using `stratify=y` to preserve class ratios

- **Feature Scaling**  
  Applying `StandardScaler` to ensure stable Gradient Descent convergence

- **Model Training**  
  Logistic Regression with `class_weight='balanced'`

- **Threshold Optimization**  
  Moving beyond the default `0.5` threshold to minimize False Negatives

- **Evaluation**  
  Confusion Matrix + Precision–Recall curve analysis

---

## 🛠️ Key Technical Decisions

### 1. Why StandardScaler?

Logistic Regression relies on Gradient Descent.

Tumor features vary greatly:

- Area can be **1000+**
- Smoothness can be **< 0.1**

Scaling is mandatory to prevent bias toward large-magnitude features.

---

### 2. Handling Class Imbalance

The dataset is imbalanced (~37% Malignant).

To correct this, the model uses:

```python
LogisticRegression(class_weight="balanced")
This forces the model to pay more attention to the minority (Malignant) class.

3. Recall > Accuracy
In medical AI:

A 95% accuracy is meaningless if the remaining 5% are missed cancer cases.

This project evaluates performance using:

Confusion Matrix

Recall Score

Precision–Recall tradeoff

Goal: Minimize False Negatives

📈 Visualizations
The script automatically generates and saves graphs inside /graphs:

Class Distribution (Benign vs Malignant)

Correlation Heatmap (Top predictive features)

Confusion Matrix (TP, TN, FP, FN breakdown)

🚀 How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/YourUsername/breast-cancer-detection.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the script:

bash
Copy code
python cancer_detection_logistic.py
🏗️ Future Work: Neural Network (V2)
This Logistic Regression model serves as the baseline.

Next phase includes:

Implementing a Deep Neural Network using TensorFlow/Keras

Comparing Recall and F1-score against this baseline

Testing whether Deep Learning reduces False Negatives significantly

Author
Abdullah
AI Engineering Student | Focused on Medical AI & Deep Learning
