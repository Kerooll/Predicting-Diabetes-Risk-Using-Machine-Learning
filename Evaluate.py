import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# ⚙️ Function to evaluate models
# ============================================================
def evaluate_models(X, y, model_prefix):
    print(f"\n🔹 Evaluating models for {model_prefix} dataset...")

    X_features = X.columns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Map model names to filenames
    model_files = {
        "Logistic Regression": f"best_Logistic_Regression_{model_prefix}.pkl",
        "Random Forest": f"best_Random_Forest_{model_prefix}.pkl",
        "XGBoost": f"best_XGBoost_{model_prefix}.pkl"
    }

    best_models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            best_models[name] = joblib.load(path)
        else:
            print(f"⚠️ Model not found: {path}")

    # Evaluate metrics
    for name, model in best_models.items():
        print(f"\nEvaluating {name} ({model_prefix})...")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        try:
            y_prob = model.predict_proba(X_test)
            if len(np.unique(y)) > 2:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            else:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            print(f"ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # --- Feature Importances / Coefficients ---
    print("\n--- Feature Importances ---")
    for name, model in best_models.items():
        # If pipeline, extract classifier
        classifier = model.named_steps['model'] if hasattr(model, 'named_steps') else model

        if name == "Random Forest" or name == "XGBoost":
            importances = pd.Series(classifier.feature_importances_, index=X_features)
            print(f"\nTop 10 Important Features ({name}):")
            print(importances.sort_values(ascending=False).head(10))
        elif name == "Logistic Regression":
            # Multi-class: take class 1 (Prediabetes) coefficients, else binary
            coef = classifier.coef_[1] if classifier.coef_.shape[0] > 1 else classifier.coef_[0]
            coef_series = pd.Series(coef, index=X_features)
            # Sort by absolute value
            coef_series = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
            print(f"\nTop 10 Coefficients ({name}):")
            print(coef_series.head(10))


# ============================================================
# 🧩 BRFSS DATASET
# ============================================================
brfss_path = "diabetes_012_health_indicators_BRFSS2015.csv"
if os.path.exists(brfss_path):
    print("\n📘 Loading BRFSS dataset...")
    df_brfss = pd.read_csv(brfss_path)

    # Fill missing values with mode
    for col in df_brfss.columns[df_brfss.isnull().any()]:
        df_brfss[col] = df_brfss[col].fillna(df_brfss[col].mode()[0])

    X_brfss = df_brfss.drop('Diabetes_012', axis=1)
    y_brfss = df_brfss['Diabetes_012']

    evaluate_models(X_brfss, y_brfss, "BRFSS_SMOTE")


# ============================================================
# 🧬 PIMA DATASET
# ============================================================
pima_path = "Pima_Indian_diabetes_Dataset.csv"
if os.path.exists(pima_path):
    print("\n📗 Loading PIMA dataset...")
    df_pima = pd.read_csv(pima_path)

    # Fill missing values with median
    if df_pima.isnull().sum().sum() > 0:
        df_pima = df_pima.fillna(df_pima.median())

    X_pima = df_pima.drop('Outcome', axis=1)
    y_pima = df_pima['Outcome']

    evaluate_models(X_pima, y_pima, "PIMA")
