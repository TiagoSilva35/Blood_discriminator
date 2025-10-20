"""
Evaluation metrics and utilities for blood type classification
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)
import time
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive evaluator for blood type classification models.
    Measures quality, efficacy, efficiency, and diversity metrics.
    """
    
    def __init__(self, class_names: Optional[list] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names or ['A', 'B', 'AB', 'O']
        self.metrics = {}
        
    def evaluate_quality(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate prediction quality metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # AUC-ROC if probabilities available
        if y_pred_proba is not None:
            try:
                # One-vs-Rest AUC
                from sklearn.preprocessing import label_binarize
                n_classes = len(np.unique(y_true))
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                metrics['auc_roc_macro'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                         average='macro', multi_class='ovr')
                metrics['auc_roc_weighted'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                            average='weighted', multi_class='ovr')
            except Exception as e:
                print(f"Could not calculate AUC-ROC: {e}")
        
        self.metrics['quality'] = metrics
        return metrics
    
    def evaluate_efficacy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model efficacy (how well it achieves its goal).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of efficacy metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Sensitivity (True Positive Rate) per class
        sensitivity = cm.diagonal() / cm.sum(axis=1)
        
        # Specificity (True Negative Rate) per class
        specificity = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        
        metrics = {
            'mean_sensitivity': np.mean(sensitivity),
            'mean_specificity': np.mean(specificity),
            'balanced_accuracy': (np.mean(sensitivity) + np.mean(specificity)) / 2
        }
        
        for i, class_name in enumerate(self.class_names):
            if i < len(sensitivity):
                metrics[f'sensitivity_{class_name}'] = sensitivity[i]
                metrics[f'specificity_{class_name}'] = specificity[i]
        
        self.metrics['efficacy'] = metrics
        return metrics
    
    def evaluate_efficiency(self, inference_time: float, 
                          n_samples: int,
                          training_time: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate model efficiency (computational performance).
        
        Args:
            inference_time: Total time for inference (seconds)
            n_samples: Number of samples processed
            training_time: Total training time (seconds, optional)
            
        Returns:
            Dictionary of efficiency metrics
        """
        metrics = {
            'total_inference_time': inference_time,
            'inference_time_per_sample': inference_time / n_samples,
            'throughput': n_samples / inference_time  # samples per second
        }
        
        if training_time is not None:
            metrics['total_training_time'] = training_time
        
        self.metrics['efficiency'] = metrics
        return metrics
    
    def evaluate_diversity(self, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate prediction diversity and confidence.
        
        Args:
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of diversity metrics
        """
        # Class distribution in predictions
        unique, counts = np.unique(y_pred, return_counts=True)
        class_distribution = dict(zip(unique, counts / len(y_pred)))
        
        # Shannon entropy of predictions
        probabilities = counts / len(y_pred)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        metrics = {
            'prediction_entropy': entropy,
            'n_unique_predictions': len(unique)
        }
        
        for class_idx, prob in class_distribution.items():
            if class_idx < len(self.class_names):
                metrics[f'pred_dist_{self.class_names[class_idx]}'] = prob
        
        if y_pred_proba is not None:
            # Average confidence
            max_probs = np.max(y_pred_proba, axis=1)
            metrics['mean_confidence'] = np.mean(max_probs)
            metrics['std_confidence'] = np.std(max_probs)
            metrics['min_confidence'] = np.min(max_probs)
            metrics['max_confidence'] = np.max(max_probs)
        
        self.metrics['diversity'] = metrics
        return metrics
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray] = None,
                                inference_time: Optional[float] = None,
                                n_samples: Optional[int] = None,
                                training_time: Optional[float] = None) -> Dict:
        """
        Perform comprehensive evaluation of all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            inference_time: Time for inference (optional)
            n_samples: Number of samples (optional)
            training_time: Training time (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Quality metrics
        results['quality'] = self.evaluate_quality(y_true, y_pred, y_pred_proba)
        
        # Efficacy metrics
        results['efficacy'] = self.evaluate_efficacy(y_true, y_pred)
        
        # Efficiency metrics
        if inference_time is not None and n_samples is not None:
            results['efficiency'] = self.evaluate_efficiency(inference_time, n_samples, training_time)
        
        # Diversity metrics
        results['diversity'] = self.evaluate_diversity(y_pred, y_pred_proba)
        
        self.metrics = results
        return results
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive text report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Report string
        """
        report_lines = ["=" * 80]
        report_lines.append("BLOOD DISCRIMINATOR - EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        for category, metrics in self.metrics.items():
            report_lines.append(f"\n{category.upper()} METRICS:")
            report_lines.append("-" * 80)
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric_name:40s}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric_name:40s}: {value}")
        
        report_lines.append("\n" + "=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


def measure_inference_time(model, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Measure inference time for a model.
    
    Args:
        model: Model with predict method
        X_test: Test data
        
    Returns:
        Tuple of (predictions, inference_time)
    """
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time
    
    return predictions, inference_time
