# src/model_trainer.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def train_baseline(self, X_train, y_train):
        """Trains Logistic Regression using liblinear for rapid convergence."""
        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=self.random_state)
        model.fit(X_train, y_train)
        return model

    def train_ensemble(self, X_train, y_train, n_estimators=100, max_depth=10):
        """Trains a tuned Random Forest Ensemble model."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

   
    def run_cross_validation(self, model, X, y, cv=5):
        """Performs Stratified 5-Fold Cross-Validation with progress prints."""
        print(f"   [CV Progress] Initializing Stratified {cv}-Fold Cross-Validation...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scoring = ['f1', 'average_precision']
        
        # We run cross_validate
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        print(f"   [CV Progress] Finished all {cv} folds successfully.")
        
        return {
            "CV_F1_Mean": np.mean(scores['test_f1']),
            "CV_F1_Std": np.std(scores['test_f1']),
            "CV_AUCPR_Mean": np.mean(scores['test_average_precision']),
            "CV_AUCPR_Std": np.std(scores['test_average_precision'])
        }