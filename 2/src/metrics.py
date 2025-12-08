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


def compare_supervised_configs(results_dict, label_names):
    '''
    in a confusion matrix, rows are true labels and columns are predicted labels
    TPR is the same as recall
    Recall (TPR) = TP / (TP + FN)
    recall for class i = CM[i,i] / sum(CM[i,:]) 
    Because CM[i, i] is the number of correct predictions for class i
    and sum(CM[i,:]) is the total number of true instances of class i

    Specificity = TN / (TN + FP)
    Where for class i:
    TN = sum of all CM elements not in row i or column i
    FP = sum of column i excluding CM[i,i]
    Thus, Specificity for class i = sum(CM[not i, not i]) / (sum(CM[not i, not i]) + sum(CM[not i, i]))

    Precision = TP / (TP + FP)
    Where for class i:
    Precision for class i = CM[i,i] / sum(CM[:,i])


    '''
    for i, label in enumerate(label_names):
        recalls = []
        specificities = []
        precisions = []
        for config in ['5_epochs', '10_epochs', '20_epochs']:
            
            cm = results_dict[config]['metrics']['confusion_matrix']

            # For Class i:
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FP + FN)

            recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            precision = TP / (TP + FP)

            recalls.append(recall)
            specificities.append(specificity)
            precisions.append(precision)
        
        # plot the metrics evolution
        print(f"Class: {label}")
        print(f"  Recall (TPR):    5 ep: {recalls[0]:.4f} | 10 ep: {recalls[1]:.4f} | 20 ep: {recalls[2]:.4f}")
        print(f"  Specificity:     5 ep: {specificities[0]:.4f} | 10 ep: {specificities[1]:.4f} | 20 ep: {specificities[2]:.4f}")
        print(f"  Precision:       5 ep: {precisions[0]:.4f} | 10 ep: {precisions[1]:.4f} | 20 ep: {precisions[2]:.4f}")
        print("")

def retrieve_top_mistakes(cm, label_names, top_n=5):
    mistakes = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            # predicted j but true i
            if i != j:
                count = cm[i, j]
                if count > 0:
                    percent = count / np.sum(cm[i, :])*100
                    mistakes.append((f"{label_names[i]} → {label_names[j]}", count, percent))
    # Sort by mistake count descending
    mistakes.sort(key=lambda x: x[2], reverse=True)
    for _ in range(top_n):
        mistake, count, percent = mistakes[_]
        print(f"  {mistake:<12} : {count:3d} ({percent:.1f}%)")
    return mistakes[:top_n]

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
