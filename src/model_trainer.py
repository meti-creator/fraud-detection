import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_secure_training_pipeline():
    # 1. DEFENSIVE LAYER 1: Hardened CSV Data Ingestion Handling
    data_path = os.path.join("notebooks", "data", "processed", "cleaned_fraud.csv")
    
    try:
        print(f"📥 Loading structural feature matrix from: {data_path}")
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ Ingestion Error: The file '{data_path}' cannot be located on disk. "
            f"Please verify that your preprocessing notebook or script has executed successfully."
        ) from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"❌ Data Integrity Error: The file at '{data_path}' is completely empty.") from e
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected OS read failure during CSV decoding: {str(e)}") from e

    # 2. DEFENSIVE LAYER 2: Structural Verification of Target and Feature Attributes
    target_column = 'class'
    if target_column not in df.columns:
        # Fallback evaluation option to look for common alternative target tags
        alternatives = ['target', 'is_fraud', 'fraud']
        found_alt = [col for col in alternatives if col in df.columns]
        if found_alt:
            target_column = found_alt[0]
            print(f"⚠️ Target column alignment shifted to detected alternative: '{target_column}'")
        else:
            raise KeyError(f"❌ Array Shape Error: Required dependent label '{target_column}' is missing from the dataset rows.")

    feature_cols = [col for col in df.columns if col.lower() not in [target_column, 'user_id', 'device_id', 'signup_time']]
    X = df[feature_cols]
    y = df[target_column]

    if X.empty or len(y) == 0:
        raise ValueError("❌ Structural Error: Split dimensions failed. Extracted feature matrix or target vector contains zero samples.")

    # Execute split matrix processing safely
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. DEFENSIVE LAYER 3: Model Fitting Validation Boundary
    try:
        print(f"🏋️ Initializing Random Forest Fit on {X_train.shape[0]} training samples across {X_train.shape[1]} features...")
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print("✅ Model convergence achieved successfully.")
    except ValueError as e:
        raise ValueError(f"❌ Mathematical Alignment Failure: Input data arrays contain illegal NaN values or incompatible datatypes: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"❌ Computational execution failure during tree compilation: {str(e)}") from e

    # 4. DEFENSIVE LAYER 4: Safe Artifact Storage & Metrics Logging
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_output_path = os.path.join(models_dir, "random_forest_fraud_model.joblib")
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")

    try:
        # Persist model binary
        joblib.dump(model, model_output_path)
        print(f"📦 Selected best model binary successfully persisted to: {model_output_path}")

        # Compute and write evaluation matrix summary json
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"💾 Metrics report securely written to: {metrics_path}")
        
    except IOError as e:
        raise IOError(f"❌ Disk Workspace Permissions Error: Unable to write serialization artifacts to disk: {str(e)}") from e

if __name__ == "__main__":
    run_secure_training_pipeline()