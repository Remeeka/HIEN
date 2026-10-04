import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'hien_dataset')
ML_DIR = os.path.join(DATA_DIR, 'ml')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading train, validation, and test datasets...")
train_df = pd.read_csv(os.path.join(ML_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(ML_DIR, 'validation.csv'))
test_df = pd.read_csv(os.path.join(ML_DIR, 'test.csv'))

feature_cols = [
    'product_category', 'unit_price', 'return_reason', 'days_since_purchase',
    'ai_condition_grade', 'ai_confidence', 'ai_fraud_risk_score',
    'has_original_packaging', 'accessories_complete', 'functional_test_result',
    'hygiene_sensitive', 'fragile', 'local_demand_score', 'previous_return_count',
    'distance_to_nearest_hub_km', 'storage_capacity_available'
]

categorical_cols = ['product_category', 'return_reason', 'ai_condition_grade', 'functional_test_result']
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

target_col = 'eligibility_label'

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

rf_clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_clf)
])

print("Training Random Forest model on 4,900 training records...")
pipeline.fit(X_train, y_train)

# Validation evaluation
y_val_pred = pipeline.predict(X_val)
y_val_prob = pipeline.predict_proba(X_val)[:, 1]

val_acc = accuracy_score(y_val, y_val_pred)
val_prec = precision_score(y_val, y_val_pred)
val_rec = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_prob)

print(f"Validation Metrics: Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")

# Test evaluation
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)
cm = confusion_matrix(y_test, y_test_pred).tolist()

print(f"Test Metrics: Acc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
print("Confusion Matrix (Test):", cm)

# Feature importance extraction
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_feature_names = numeric_cols + cat_feature_names

importances = pipeline.named_steps['classifier'].feature_importances_
top_indices = np.argsort(importances)[::-1][:15]

top_features = [
    {"feature": all_feature_names[i], "importance": round(float(importances[i]), 4)}
    for i in top_indices
]

metadata = {
    "model_type": "Random Forest Classifier Pipeline",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "metrics": {
        "accuracy": round(float(test_acc), 4),
        "precision": round(float(test_prec), 4),
        "recall": round(float(test_rec), 4),
        "f1_score": round(float(test_f1), 4),
        "roc_auc": round(float(test_auc), 4),
        "confusion_matrix": {
            "true_negative": cm[0][0],
            "false_positive": cm[0][1],
            "false_negative": cm[1][0],
            "true_positive": cm[1][1]
        }
    },
    "feature_cols": feature_cols,
    "top_feature_importances": top_features
}

model_path = os.path.join(MODEL_DIR, 'eligibility_model.joblib')
meta_path = os.path.join(MODEL_DIR, 'model_metadata.json')

joblib.dump(pipeline, model_path)
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model successfully saved to {model_path}")
print(f"Metadata successfully saved to {meta_path}")
