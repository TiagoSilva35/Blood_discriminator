from pathlib import Path


def save_logs(results, filepath="logs/results.txt"):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    """
    the results has the structure:
    {'X_epochs': {'model': ANNClassifier(
  (net): Sequential(
    (0): Linear(in_features=384, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=256, out_features=6, bias=True)
  )
), 'metrics': {'accuracy': double, 'macro_f1': double, 'confusion_matrix': array([[...],
       [...],
       [...],
       [...],
       [...],
       [...]]), 
    'classification_report': 'precision    recall  f1-score   support\n\n     
    sadness...\n         
    joy...\n        
    love...\n       
    anger...\n        
    fear...\n    
    surprise...\n\n    
    accuracy                           0.70      2000\n   
    macro avg       0.64      0.59      0.60      2000\n
    weighted avg       0.69      0.70      0.69      2000\n'}, 
    'label_names': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']}
    """

    with open(filepath, "w") as f:
        for model, metrics in results.items():
            f.write(f"Model: {model} specs: {results[model]['model']}\n")
            f.write(f"Accuracy: {metrics['metrics']['accuracy']:.4f}\n")
            f.write(f"Macro F1: {metrics['metrics']['macro_f1']:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(f"{metrics['metrics']['confusion_matrix']}\n")
            f.write("Classification Report:\n")
            f.write(f"{metrics['metrics']['classification_report']}\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"Results saved to {filepath}")
