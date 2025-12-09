import torch
from transformers import pipeline
from tqdm.auto import tqdm

from metrics import compute_metrics, print_metrics


# ---------------------------------------------------------
# Configurations
# ---------------------------------------------------------

# zero-Shot Models to be compared
ZERO_SHOT_MODELS = [
    "facebook/bart-large-mnli",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------


def evaluate_single_zero_shot_model(model_name, label_names, X_test_texts, y_test):
    print(f"\nEvaluating Zero-Shot Model: {model_name}")
    print(f"Loading pipeline...")

    device_id = (
        0 if torch.cuda.is_available() and str(DEVICE).startswith("cuda") else -1
    )

    try:
        zero_shot = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device_id,
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

    preds = []
    print("Making predictions...")
    for text in tqdm(X_test_texts, desc=f"Zero-shot ({model_name})"):
        try:
            result = zero_shot(text, candidate_labels=label_names)
            pred_label = result["labels"][0]
            preds.append(label_names.index(pred_label))
        except Exception as e:
            print(f"Error in prediction: {e}")
            preds.append(0)

    # compute metrics
    metrics = compute_metrics(y_test, preds, label_names)

    return metrics


def evaluate_zero_shot_models(X_test_texts, y_test, label_names):
    print(f"  Zero-Shot Emotion Classification")
    print(f"Models to evaluate: {ZERO_SHOT_MODELS}")
    print(f"Using device: {DEVICE}")

    results = {}

    for model_name in ZERO_SHOT_MODELS:
        metrics = evaluate_single_zero_shot_model(
            model_name, label_names, X_test_texts, y_test
        )

        if metrics is not None:
            # display results
            model_short_name = model_name.split("/")[-1]
            print_metrics(
                metrics, label_names, model_name=f"Zero-Shot: {model_short_name}"
            )

            # store results
            results[model_name] = {"metrics": metrics, "label_names": label_names}
        else:
            print(f"Skipping model {model_name} due to errors.")

    # final Comparison
    print(f"  Zero-Shot Models Comparison Summary")
    print(f"{'Model':<50} {'Accuracy':<12} {'Macro-F1':<12}")

    for model_name, result in results.items():
        model_short = model_name.split("/")[-1][:48]
        acc = result["metrics"]["accuracy"]
        f1 = result["metrics"]["macro_f1"]
        print(f"{model_short:<50} {acc:<12.4f} {f1:<12.4f}")

    return results
