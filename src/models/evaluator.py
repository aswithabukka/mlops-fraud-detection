"""Model evaluation and metrics visualization."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate and visualize model performance."""

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')

        if save_path:
            plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, save_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path)
        plt.close()
