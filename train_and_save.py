import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import warnings

# --- Import SMOTE and the imblearn Pipeline ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 🚫 BRFSS 2015 DATASET SECTION (with SMOTE)
# ============================================================

print("🔹 Loading BRFSS 2015 dataset...")
df_brfss = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Handle missing values (Mode Imputation)
cols_with_missing = df_brfss.columns[df_brfss.isnull().any()].tolist()
df_brfss_imputed = df_brfss.copy()
for col in cols_with_missing:
    mode_val = df_brfss_imputed[col].mode()[0]
    df_brfss_imputed[col] = df_brfss_imputed[col].fillna(mode_val)
df_brfss = df_brfss_imputed

# Features and target
X_brfss = df_brfss.drop('Diabetes_012', axis=1)
y_brfss = df_brfss['Diabetes_012']

X_train, X_test, y_train, y_test = train_test_split(
    X_brfss, y_brfss, test_size=0.2, random_state=42, stratify=y_brfss
)
# Note: Added stratify=y_brfss to ensure both train and test sets
# have the same (imbalanced) class distribution.

print(f"✅ BRFSS dataset ready. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ---
# Define models and parameters for BRFSS using Pipelines
# ---

models_brfss = {
    'Logistic Regression': {
        'pipeline': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'))
        ]),
        'param_grid': {
            'model__C': [0.1, 1, 10], # Note the 'model__' prefix
            'model__penalty': ['l2']
        }
    },
    'Random Forest': {
        'pipeline': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        'param_grid': {
            'model__n_estimators': [100, 200], # Note the 'model__' prefix
            'model__max_depth': [10, 20],
            'model__min_samples_split': [5, 10]
        }
    },
    'XGBoost': {
        'pipeline': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', XGBClassifier(objective='multi:softmax', use_label_encoder=False, 
                                   eval_metric='mlogloss', random_state=42))
        ]),
        'param_grid': {
            'model__n_estimators': [100, 200], # Note the 'model__' prefix
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1]
        }
    }
}

best_models_brfss = {}

for name, model_info in models_brfss.items():
    print(f"\n🚀 Performing GridSearchCV for {name} (BRFSS with SMOTE)...")
    
    # Use 'f1_weighted' to optimize for imbalanced classes, not just 'accuracy'
    grid_search = GridSearchCV(model_info['pipeline'], model_info['param_grid'],
                               cv=5, scoring='f1_weighted', n_jobs=-1) 
    # Reduced CV to 5 for faster training with SMOTE
    
    grid_search.fit(X_train, y_train)
    
    best_models_brfss[name] = grid_search.best_estimator_
    print(f"✅ Best parameters for {name}: {grid_search.best_params_}")
    print(f"⭐ Best CV (f1_weighted) score for {name}: {grid_search.best_score_:.4f}")
    
    # Save the *entire pipeline* (which includes SMOTE + the trained model)
    joblib.dump(grid_search.best_estimator_, f"best_{name.replace(' ', '_')}_BRFSS_SMOTE.pkl")

print("\n🎉 BRFSS (SMOTE) models trained and saved successfully!")

# ============================================================
# ✅ PIMA INDIAN DATASET SECTION
# ============================================================

print("\n🔹 Loading PIMA Indian Diabetes dataset...")
df_pima = pd.read_csv("Pima_Indian_diabetes_Dataset.csv")

# Handle missing values (Median Imputation)
if df_pima.isnull().sum().sum() > 0:
    print("⚠️ Missing values detected — filling with median values.")
    df_pima = df_pima.fillna(df_pima.median())

# Split features and target
X_pima = df_pima.drop('Outcome', axis=1)
y_pima = df_pima['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=42
)

print(f"✅ PIMA dataset ready. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Define models for PIMA (binary classification)
models_pima = {
    'Logistic Regression': {
        'model': LogisticRegression(solver='liblinear', max_iter=1000),
        'param_grid': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [100, 200], 'max_depth': [5, 10, 20], 
                       'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    },
    'XGBoost': {
        'model': XGBClassifier(objective='binary:logistic',
                               eval_metric='logloss', random_state=42),
        'param_grid': {'n_estimators': [100, 200], 'max_depth': [3, 6], 
                       'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1.0]}
    }
}

best_models_pima = {}

for name, model_info in models_pima.items():
    print(f"\n🚀 Performing GridSearchCV for {name} (PIMA)...")
    grid_search = GridSearchCV(model_info['model'], model_info['param_grid'],
                               cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models_pima[name] = grid_search.best_estimator_
    print(f"✅ Best parameters for {name}: {grid_search.best_params_}")
    print(f"⭐ Best CV score for {name}: {grid_search.best_score_:.4f}")
    joblib.dump(grid_search.best_estimator_, f"best_{name.replace(' ', '_')}_PIMA.pkl")


rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42
)

cv_auc = cross_val_score(rf, X_pima, y_pima, cv=5, scoring='roc_auc')
print(f"RF ROC AUC (5-CV): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

print("\n🎉 All PIMA models trained and saved successfully!")
