"""
Módulo de métricas para avaliação de modelos de classificação.
Contém funções para calcular Accuracy, Macro-F1, Confusion Matrix e Classification Report.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, label_names):
    """
    Calcula todas as métricas exigidas para o projeto.
    
    Args:
        y_true: Array com os rótulos verdadeiros
        y_pred: Array com os rótulos preditos
        label_names: Lista com os nomes das classes
    
    Returns:
        dict: Dicionário contendo:
            - 'accuracy': Acurácia
            - 'macro_f1': Macro F1-Score
            - 'confusion_matrix': Matriz de confusão (valores absolutos)
            - 'classification_report': Relatório de classificação completo
    """
    # Calcular métricas
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'classification_report': report
    }


def print_metrics(metrics, label_names, model_name="Model"):
    """
    Imprime as métricas de forma formatada.
    
    Args:
        metrics: Dicionário retornado pela função compute_metrics
        label_names: Lista com os nomes das classes
        model_name: Nome do modelo para exibição
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} - Test Results")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"Labels: {label_names}")
    print(metrics['confusion_matrix'])
    print(f"\nClassification Report:")
    print(metrics['classification_report'])
    print(f"{'='*60}\n")
