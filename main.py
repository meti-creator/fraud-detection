# main.py
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
import os

def run_pipeline(data_path, target_column, dataset_name):
    print(f"\n=====================================================================")
    print(f"RUNNING EXPERIMENT FOR: {dataset_name}")
    print(f"=====================================================================")
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}. Skipping...")
        return
        
    # 1. Load and Split
    preprocessor = DataPreprocessor(data_path, target_column)
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # 2. Train and Cross-Validate Baseline
    print("-> Training & Cross-Validating Logistic Regression...")
    lr_model = trainer.train_baseline(X_train, y_train)
    lr_metrics = evaluator.evaluate(lr_model, X_test, y_test)
    lr_cv = trainer.run_cross_validation(lr_model, X_train, y_train)
    
    # 3. Train and Cross-Validate Ensemble
    print("-> Training & Cross-Validating Random Forest...")
    rf_model = trainer.train_ensemble(X_train, y_train, n_estimators=100, max_depth=10)
    rf_metrics = evaluator.evaluate(rf_model, X_test, y_test)
    rf_cv = trainer.run_cross_validation(rf_model, X_train, y_train)
    
    # 4. Generate Combined Metrics Report Table
    report_df = evaluator.generate_report(lr_metrics, lr_cv, rf_metrics, rf_cv)
    print("\nFINAL METRICS TABLE:")
    print(report_df.to_string(index=False))

def main():
    # Dataset 1 Execution
    run_pipeline(
        data_path="C:\\Users\\HP\\fraud-detection\\data\\raw\\creditcard.csv", 
        target_column="Class", 
        dataset_name="Credit Card Transactions (creditcard.csv)"
    )
    
    # Dataset 2 Execution
    run_pipeline(
        data_path="C:\\Users\\HP\\fraud-detection\\data\\raw\\Fraud_Data.csv", 
        target_column="class", 
        dataset_name="E-commerce Activity Logs (Fraud_Data.csv)"
    )

if __name__ == "__main__":
    main()