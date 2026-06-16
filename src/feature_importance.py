import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_and_plot_feature_importance(model_path="models/random_forest_fraud_model.joblib", 
                                        feature_names_path="data/processed/feature_names.joblib"):
    """
    Safely extracts built-in feature importances from a trained ensemble model,
    validates structural constraints, and saves a top-10 visualization to disk.
    """
    print("🚀 Initializing Feature Importance Baseline Pipeline...")
    
    # 1. DEFENSIVE ERROR HANDLING: Verify file paths exist
    if not os.path.exists(model_path):
        print(f"❌ CONFIGURATION ERROR: Saved model artifact not found at '{model_path}'.")
        print("Please ensure your Task 2 model training script has run successfully.")
        sys.exit(1)
        
    try:
        # Load the serialized model
        model = joblib.load(model_path)
        print("Model artifact successfully loaded into memory.")
    except Exception as e:
        print(f"INGESTION FAILURE: Failed to deserialize model file: {str(e)}")
        sys.exit(1)
        
    # 2. VALIDATION: Ensure the model has built-in feature importances
    if not hasattr(model, "feature_importances_"):
        print("❌ VALIDATION ERROR: The loaded model does not support 'feature_importances_'.")
        print("Ensure you are using an ensemble method like Random Forest or XGBoost.")
        sys.exit(1)
        
    # 3. EXTRACT AND MAPPED FEATURES
    try:
        importances = model.feature_importances_
        
        # Load feature names if they exist, otherwise generate placeholder names
        # Explicit feature list matching the 9 columns trained in modeling.py
        feature_names = [
            'purchase_value', 'age', 'source_Direct', 'source_SEO', 
            'browser_FireFox', 'browser_IE', 'browser_Opera', 
            'browser_Safari', 'sex_M'
        ]
        
        # Guard check to ensure dimension alignment
        if len(feature_names) != len(importances):
            print(f"⚠️ Warning: Feature length mismatch. Falling back to generic indexing.")
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
        # Create a structured DataFrame for clean sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Isolate the top 10 most impactful features
        top_10_features = importance_df.head(10).copy()
        
    except Exception as e:
        print(f"❌ PROCESSING ERROR: Failed to map feature importances: {str(e)}")
        sys.exit(1)
        
    # 4. PLOT VISUALIZATION AND SAVE ARTIFACT
    try:
        # Ensure target directories exist for report outputs
        os.makedirs("reports/figures", exist_ok=True)
        output_image_path = "reports/figures/baseline_feature_importance.png"
        
        plt.figure(figsize=(10, 6))
        # Invert the order so the most important feature is on top of the horizontal bar chart
        plt.barh(top_10_features['Feature'][::-1], top_10_features['Importance'][::-1], color='dodgerblue')
        plt.xlabel('Relative Mathematical Importance')
        plt.title('Top 10 Most Predictive Behavioral Features (Minitab/Ensemble Baseline)')
        plt.tight_layout()
        
        # Save plot as a clean PNG image for the report
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        
        print(f" SUCCESS: Top 10 Feature Importance plot saved to: {output_image_path}")
        
        # Log textual output for verification
        print("\n--- Top 5 Features Log Breakdown ---")
        for idx, row in top_10_features.head(5).iterrows():
            print(f"Rank {idx+1}: {row['Feature']} ({row['Importance']:.4f})")
            
    except Exception as e:
        print(f"❌ VISUALIZATION FAILURE: Could not compile or save plot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    extract_and_plot_feature_importance()