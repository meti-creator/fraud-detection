# src/evaluator.py
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import pandas as pd

class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        """Calculates holdout evaluation scores."""
        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        return {"AUC-PR": auc_pr, "F1-Score": f1, "Confusion Matrix": cm}

    @staticmethod
    def generate_report(lr_metrics, lr_cv, rf_metrics, rf_cv):
        """Creates the comprehensive summary table required by instructions."""
        data = {
            "Metric": [
                "Holdout AUC-PR", 
                "5-Fold CV AUC-PR (Mean ± Std)", 
                "Holdout F1-Score", 
                "5-Fold CV F1-Score (Mean ± Std)",
                "Confusion Matrix (TN / FP / FN / TP)"
            ],
            "Logistic Regression (Baseline)": [
                f"{lr_metrics['AUC-PR']:.4f}",
                f"{lr_cv['CV_AUCPR_Mean']:.4f} ± {lr_cv['CV_AUCPR_Std']:.4f}",
                f"{lr_metrics['F1-Score']:.4f}",
                f"{lr_cv['CV_F1_Mean']:.4f} ± {lr_cv['CV_F1_Std']:.4f}",
                f"{lr_metrics['Confusion Matrix'].ravel()}"
            ],
            "Random Forest (Ensemble)": [
                f"{rf_metrics['AUC-PR']:.4f}",
                f"{rf_cv['CV_AUCPR_Mean']:.4f} ± {rf_cv['CV_AUCPR_Std']:.4f}",
                f"{rf_metrics['F1-Score']:.4f}",
                f"{rf_cv['CV_F1_Mean']:.4f} ± {rf_cv['CV_F1_Std']:.4f}",
                f"{rf_metrics['Confusion Matrix'].ravel()}"
            ]
        }
        return pd.DataFrame(data)