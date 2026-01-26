# Breast Cancer Detection using Logistic Regression

## Objective
Build a clean machine learning baseline model to classify breast tumors as:

- **0 = Malignant (Cancer)**
- **1 = Benign (Healthy)**

This project demonstrates a full medical classification pipeline using **Logistic Regression**.

---

## What this project demonstrates

- How Logistic Regression works for binary cancer prediction  
- Why medical datasets require careful evaluation (not just accuracy)  
- How feature scaling improves model performance  
- How confusion matrices reveal real diagnostic errors  
- Why threshold tuning matters in healthcare

---

## Workflow: Features → Model → Prediction

---

## 1. Load Breast Cancer Dataset (Real Medical Data)

We use the built-in **Wisconsin Breast Cancer Dataset** from scikit-learn.

- 569 patient samples  
- 30 diagnostic features  
- Binary target (benign vs malignant)

---

## 2. Class Balance Check (Medical Reality)

Medical datasets are often **imbalanced**.

Class distribution:

- Benign (Healthy): 357  
- Malignant (Cancer): 212  

Why this matters:

- A biased model may ignore the minority cancer class  
- False negatives are dangerous in diagnosis  

---

## 3. Feature Correlation (Top Predictive Symptoms)

We visualize the most correlated features using a heatmap.

This helps identify:

- Which tumor measurements contribute most to prediction  
- Strong relationships between key medical features  

---

## 4. Train–Test Split

We split the dataset into:

- 80% Training  
- 20% Testing  

Using `stratify=y` to preserve class balance.

---

## 5. Feature Scaling (Critical Step)

We apply **StandardScaler** because Logistic Regression is sensitive to feature magnitude.

Scaling ensures:

- Faster convergence  
- Fair contribution of all features  

---

## 6. Logistic Regression Model (Baseline)

We train a Logistic Regression classifier with:

- `class_weight="balanced"` for medical fairness  
- Increased iterations for stability  

This provides a strong baseline before deep learning.

---

## 7. Prediction + Probabilities

The model outputs:

- Class predictions (0 or 1)  
- Probability scores for threshold control  

Probabilities are essential for ROC and threshold tuning.

---

## 8. Evaluation Metrics (What Matters in Healthcare)

We evaluate using:

- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  

Because accuracy alone is not enough in cancer detection.

---

## 9. Confusion Matrix Heatmap (Error Analysis)

The confusion matrix shows:

- Correct benign predictions  
- Correct malignant predictions  
- False positives  
- False negatives (most critical)

---

## 10. Precision–Recall Curve

We plot Precision vs Recall to understand:

- Sensitivity to cancer cases  
- Tradeoff between missed cancer and false alarms  

More useful than ROC in imbalanced medical data.

---

## 11. Threshold Experiment (Reducing False Negatives)

Instead of default threshold = 0.5, we test:

- Threshold = 0.3

Goal:

- Reduce missed cancer cases  
- Increase recall for malignant detection  

This is a realistic healthcare adjustment.

---

## Results Summary

This project builds a complete medical ML classification baseline with:

- Logistic Regression  
- Proper preprocessing  
- Strong evaluation  
- Graph-based interpretability  
- Threshold control for safety  

---

## Next Improvements

- Add Neural Network model for comparison  
- Hyperparameter tuning  
- Cross-validation  
- Deployment as a medical prediction app

---

📌 **Repo includes saved graphs inside `/images/` folder.**
