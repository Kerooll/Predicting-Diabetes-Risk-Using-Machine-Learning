# 🧪 Diabetes Risk Prediction System  
🚀 Machine Learning | 📊 Data Analytics | 🧠 Healthcare AI  

---

## 📌 Overview  

This project presents a **machine learning-based diabetes risk prediction system** developed as a Final Year Project.  
It focuses on **early detection and awareness** by leveraging both **lifestyle data** and **clinical health indicators**.

The system integrates:
- Predictive machine learning models  
- Feature importance analysis  
- Interactive dashboard for real-time predictions  

---

## 📊 Datasets Used  

### 🔹 BRFSS 2015 Dataset  
- ~250,000 records (large-scale real-world dataset)  
- Lifestyle & behavioral indicators (BMI, physical activity, smoking, etc.)  
- Multi-class classification:  
  - 0 → No Diabetes  
  - 1 → Prediabetes  
  - 2 → Diabetes  

### 🔹 PIMA Indian Diabetes Dataset  
- Clinical dataset  
- Features: Glucose, BMI, Insulin, Age, etc.  
- Binary classification:  
  - 0 → No Diabetes  
  - 1 → Diabetes  

---

## ⚙️ Key Features  

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

- **Logistic Regression + SMOTE** improved minority class detection (prediabetes)  
- **Random Forest** achieved highest performance on PIMA dataset  
- BRFSS (lifestyle data) is useful for **large-scale screening**  
- PIMA (clinical data) provides **more precise predictions**  

---

## ⚠️ Model Files  

Pre-trained `.pkl` model files are **not included** due to GitHub size limitations.

To generate trained models locally, run:

```bash
python train_and_save.py
