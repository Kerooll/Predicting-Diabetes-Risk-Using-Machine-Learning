# 🧪 Diabetes Risk Prediction System

A machine learning-based system designed to predict diabetes risk using both lifestyle and clinical datasets. This project aims to support early detection and raise awareness through an interactive dashboard.

---

## 📌 Overview

Diabetes is a growing global health concern. Early detection is critical but often limited by access to clinical testing. This project leverages machine learning to predict diabetes risk using non-invasive and clinical indicators.

The system integrates:
- Predictive modeling (ML algorithms)
- Feature importance analysis
- Interactive dashboard for real-time predictions

---

## 📊 Datasets Used

### 1. BRFSS 2015 Dataset
- ~250,000 records
- Lifestyle & health indicators (BMI, physical activity, smoking, etc.)
- Multi-class classification:
  - 0: No diabetes
  - 1: Prediabetes
  - 2: Diabetes

### 2. PIMA Indian Diabetes Dataset
- Clinical dataset
- Features include glucose, insulin, BMI, age, etc.
- Binary classification:
  - 0: No diabetes
  - 1: Diabetes

---

## ⚙️ Features

- Data preprocessing (missing values, encoding, normalization)
- Class imbalance handling using **SMOTE**
- Machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation:
  - Accuracy, Precision, Recall, F1-score
  - ROC Curve & AUC
  - Confusion Matrix
- Feature importance analysis
- Interactive **Streamlit dashboard**

---

## 🧠 Key Insights

- **Logistic Regression + SMOTE** performed best for handling class imbalance in BRFSS
- **Random Forest** achieved highest performance on PIMA dataset
- Lifestyle data (BRFSS) is noisier but useful for large-scale screening
- Clinical data (PIMA) provides more precise predictions

---

## 🖥️ Dashboard Features

- Model comparison (BRFSS vs PIMA)
- ROC curves for all models
- Confusion matrices visualization
- Risk prediction tool (PIMA-based)
- Health factor explanations based on user input

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/diabetes-risk-prediction.git
cd diabetes-risk-prediction
