# Credit Card Fraud Detection using Machine Learning

This project implements an end-to-end **Credit Card Fraud Detection** system using machine learning techniques on a highly imbalanced real-world dataset.

The goal is to accurately identify fraudulent transactions while minimizing false positives, a critical requirement in real-world financial systems.

---

## 🚀 Project Overview

- **Problem Type:** Binary Classification (Fraud Detection)
- **Domain:** Finance / Machine Learning
- **Dataset:** European Cardholders Credit Card Transactions
- **Challenge:** Extreme class imbalance (0.17% fraud cases)

---

## 📊 Dataset Information

- Total transactions: **284,807**
- Fraudulent transactions: **492**
- Legitimate transactions: **284,315**
- Features are PCA-transformed (`V1`–`V28`) for privacy protection

📎 Dataset Source:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧠 Machine Learning Approach

The project follows a structured ML pipeline:

1. Data loading and inspection  
2. Train-test split with stratification  
3. Baseline Logistic Regression model  
4. Handling class imbalance using **SMOTE**  
5. Training and evaluation of:
   - Logistic Regression (Baseline)
   - Logistic Regression + SMOTE
   - Random Forest with class weighting
6. Model evaluation using:
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - ROC-AUC Score

---

## 📈 Model Performance Summary

### 🔹 Baseline Logistic Regression
- Fraud Recall: **0.68**
- High accuracy but misses many fraud cases

### 🔹 Logistic Regression + SMOTE
- Fraud Recall: **0.90**
- Improved fraud detection
- Higher false positives

### 🔹 Random Forest (Final Model)
- Fraud Recall: **0.76**
- Fraud Precision: **0.96**
- ROC-AUC: **~0.96**
- Best balance between recall and precision

➡️ **Random Forest was selected as the final model** due to its strong real-world performance.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 🎯 Key Learnings

- Accuracy is misleading for imbalanced datasets

- Recall is critical in fraud detection systems

- SMOTE improves recall but increases false positives

- Tree-based models handle imbalance more effectively

- ROC-AUC is a better evaluation metric than accuracy alone