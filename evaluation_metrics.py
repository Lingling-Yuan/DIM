# evaluation_metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics based on true and predicted labels:
      - Accuracy
      - Precision (macro)
      - Recall (macro)
      - F1-score (macro)
      - G-mean: Geometric mean of per-class recalls (sensitivity)
      - MCC: Matthews Correlation Coefficient
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Compute confusion matrix to derive per-class recall (sensitivity)
    cm = confusion_matrix(y_true, y_pred)
    class_recalls = []
    for i in range(cm.shape[0]):
        total = np.sum(cm[i])
        rec = cm[i, i] / total if total > 0 else 0.0
        class_recalls.append(rec)
    
    # For multiclass classification, G-mean is defined as the Nth root of the product of per-class recalls
    if all(r > 0 for r in class_recalls):
        gmean = np.prod(class_recalls) ** (1.0 / len(class_recalls))
    else:
        gmean = 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'gmean': gmean,
        'mcc': mcc
    }
    return metrics
