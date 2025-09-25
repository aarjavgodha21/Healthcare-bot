import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


DATA_FILE = "diagno_genie_demo_10000.csv"
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "diagnogenie_model.pkl")
PREPROCESSOR_FILE = os.path.join(BASE_DIR, "preprocessor.pkl")
FEATURE_CONFIG_FILE = os.path.join(BASE_DIR, "feature_config.json")


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], str]:
    """Infer feature types from the dataset."""
    
    candidate_text_cols = [c for c in df.columns if "symptom" in c.lower() or "text" in c.lower()]
    text_col = candidate_text_cols[0] if candidate_text_cols else "symptoms_text"
    if text_col not in df.columns:
        raise ValueError(
            f"Could not find a text column. Expected 'symptoms_text' or one of {candidate_text_cols}. Columns present: {list(df.columns)}"
        )

    # Exclude ID columns and dates from features
    exclude_cols = {target_col, text_col, "patient_id", "admission_date", "xray_filename"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Define which columns should be categorical (even if they look numeric)
    categorical_override = {"sex", "has_xray"}
    
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col in categorical_override:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's actually categorical (few unique values)
            if df[col].nunique() <= 10 and df[col].dtype in ['int64', 'int32']:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, text_col


def preprocess_features(X_train, X_test, numeric_cols, categorical_cols, text_col):
    """Preprocess features separately to avoid dimension issues."""
    
    processed_features = []
    
    # Process numeric features
    if numeric_cols:
        print(f"Processing {len(numeric_cols)} numeric features...")
        numeric_imputer = SimpleImputer(strategy="median")
        numeric_scaler = StandardScaler()
        
        X_train_num = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_test_num = numeric_imputer.transform(X_test[numeric_cols])
        
        X_train_num = numeric_scaler.fit_transform(X_train_num)
        X_test_num = numeric_scaler.transform(X_test_num)
        
        processed_features.append(('numeric', X_train_num, X_test_num, numeric_imputer, numeric_scaler))
    
    # Process categorical features
    if categorical_cols:
        print(f"Processing {len(categorical_cols)} categorical features...")
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
        
        X_train_cat = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_test_cat = categorical_imputer.transform(X_test[categorical_cols])
        
        X_train_cat = categorical_encoder.fit_transform(X_train_cat)
        X_test_cat = categorical_encoder.transform(X_test_cat)
        
        processed_features.append(('categorical', X_train_cat, X_test_cat, categorical_imputer, categorical_encoder))
    
    # Process text features
    if text_col:
        print(f"Processing text feature: {text_col}")
        text_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        X_train_text = text_vectorizer.fit_transform(X_train[text_col].fillna(""))
        X_test_text = text_vectorizer.transform(X_test[text_col].fillna(""))
        
        processed_features.append(('text', X_train_text, X_test_text, None, text_vectorizer))
    
    # Combine all features
    print("Combining all features...")
    X_train_combined = hstack([features[1] for features in processed_features])
    X_test_combined = hstack([features[2] for features in processed_features])
    
    # Convert to dense if needed
    if hasattr(X_train_combined, "toarray"):
        X_train_combined = X_train_combined.toarray()
    if hasattr(X_test_combined, "toarray"):
        X_test_combined = X_test_combined.toarray()
    
    return X_train_combined, X_test_combined, processed_features


def main() -> None:
    """Main training function."""
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file '{DATA_FILE}' not found in current directory: {os.getcwd()}")

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    target_col_candidates = [c for c in df.columns if c.lower() == "diagnosis_label"]
    if not target_col_candidates:
        raise ValueError("Target column 'diagnosis_label' not found in dataset.")
    target_col = target_col_candidates[0]

    numeric_cols, categorical_cols, text_col = infer_feature_types(df, target_col)
    
    print(f"Detected columns:")
    print(f"  Numeric: {numeric_cols}")
    print(f"  Categorical: {categorical_cols}")
    print(f"  Text: {text_col}")
    print(f"  Target: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Target distribution:")
    print(y.value_counts())

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Preprocess features
    X_train_processed, X_test_processed, preprocessors = preprocess_features(
        X_train, X_test, numeric_cols, categorical_cols, text_col
    )
    
    print(f"Processed feature matrix shape: {X_train_processed.shape}")

    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight=None,
    )

    model.fit(X_train_processed, y_train)

    print("Making predictions...")
    y_pred = model.predict(X_test_processed)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_processed)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    print("\nSaving model and preprocessors...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessors, PREPROCESSOR_FILE)

    feature_config = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "text_feature": text_col,
        "target": target_col,
        "classes": model.classes_.tolist(),
    }
    with open(FEATURE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    print(f"\nSaved model to {MODEL_FILE}, preprocessors to {PREPROCESSOR_FILE}, and config to {FEATURE_CONFIG_FILE}.")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()