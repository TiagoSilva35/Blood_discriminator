import sys
from pathlib import Path

# Adicionar diretório src ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from supervised import train_and_evaluate_supervised
from zero_shot import evaluate_zero_shot_models


def main():
    print("PART 1: SUPERVISED LEARNING")
    supervised_results, test_data = train_and_evaluate_supervised()
    X_test_texts, y_test, label_names = test_data
    
    print("PART 2: ZERO-SHOT LEARNING")

    zero_shot_results = evaluate_zero_shot_models(X_test_texts, y_test, label_names)
    print("FINAL COMPARISON")
    # Supervised results
    for config_name, result in supervised_results.items():
        approach = f"Supervised ANN ({config_name.replace('_', ' ')})"
        acc = result['metrics']['accuracy']
        f1 = result['metrics']['macro_f1']
        print(f"{approach:<55} {acc:<12.4f} {f1:<12.4f}")
    
    # Zero-shot results
    for model_name, result in zero_shot_results.items():
        approach = f"Zero-Shot: {model_name.split('/')[-1][:40]}"
        acc = result['metrics']['accuracy']
        f1 = result['metrics']['macro_f1']
        print(f"{approach:<55} {acc:<12.4f} {f1:<12.4f}")

if __name__ == "__main__":
    main()
