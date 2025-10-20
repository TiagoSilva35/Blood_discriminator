"""
Main training script for Blood Discriminator
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.data_loader import BloodDataPreprocessor, split_data
from models.classifier import CNNBloodClassifier, TraditionalMLClassifier
from evaluation.metrics import ModelEvaluator, measure_inference_time
from utils.helpers import load_config, save_results, ensure_dir


def train_cnn_model(config: dict, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train CNN model."""
    print("\n" + "="*80)
    print("Training CNN Model")
    print("="*80)
    
    # Initialize model
    model = CNNBloodClassifier(
        input_shape=config['model']['input_shape'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['training']['learning_rate']
    )
    
    # Train
    import time
    start_time = time.time()
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size']
    )
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred, inference_time = measure_inference_time(model, X_test)
    y_pred_proba = model.model.predict(X_test)
    
    evaluator = ModelEvaluator(class_names=config['data']['class_names'])
    results = evaluator.comprehensive_evaluation(
        y_test, y_pred, y_pred_proba,
        inference_time=inference_time,
        n_samples=len(X_test),
        training_time=training_time
    )
    
    # Save model
    model_path = Path(config['paths']['models_dir']) / 'cnn_model.h5'
    ensure_dir(config['paths']['models_dir'])
    model.save_model(str(model_path))
    
    # Generate report
    report = evaluator.generate_report(
        save_path=str(Path(config['paths']['results_dir']) / 'cnn_evaluation.txt')
    )
    print(report)
    
    # Save confusion matrix
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(Path(config['paths']['results_dir']) / 'cnn_confusion_matrix.png')
    )
    
    return results


def train_traditional_model(config: dict, model_type: str, 
                           X_train, X_val, X_test, y_train, y_val, y_test):
    """Train traditional ML model."""
    print("\n" + "="*80)
    print(f"Training {model_type.upper()} Model")
    print("="*80)
    
    # Initialize model
    model_params = config['traditional_models'].get(model_type, {})
    model = TraditionalMLClassifier(model_type=model_type, **model_params)
    
    # Combine train and val for traditional models
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Train
    import time
    start_time = time.time()
    model.train(X_train_full, y_train_full)
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred, inference_time = measure_inference_time(model, X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    evaluator = ModelEvaluator(class_names=config['data']['class_names'])
    results = evaluator.comprehensive_evaluation(
        y_test, y_pred, y_pred_proba,
        inference_time=inference_time,
        n_samples=len(X_test),
        training_time=training_time
    )
    
    # Generate report
    report = evaluator.generate_report(
        save_path=str(Path(config['paths']['results_dir']) / f'{model_type}_evaluation.txt')
    )
    print(report)
    
    # Save confusion matrix
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(Path(config['paths']['results_dir']) / f'{model_type}_confusion_matrix.png')
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Blood Discriminator Models')
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'cnn', 'random_forest', 'svm', 'gradient_boosting', 'mlp'],
                       help='Model to train')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    
    # Ensure directories exist
    for path_key in ['models_dir', 'results_dir']:
        ensure_dir(config['paths'][path_key])
    
    print("="*80)
    print("BLOOD DISCRIMINATOR - TRAINING PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("Loading and Preprocessing Data")
    print("="*80)
    
    # For demonstration, create synthetic data
    # In real usage, this would load actual blood sample images
    print("\nNote: Using synthetic data for demonstration.")
    print("Replace this with actual blood sample image loading in production.")
    
    n_samples = config['data'].get('n_samples', 1000)
    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(n_samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, num_classes, n_samples)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_state=config['data']['random_state']
    )
    
    print(f"\nData splits:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")
    
    # Train models
    all_results = {}
    
    if args.model == 'all' or args.model == 'cnn':
        results = train_cnn_model(config, X_train, X_val, X_test, y_train, y_val, y_test)
        all_results['cnn'] = results
    
    if args.model == 'all':
        traditional_models = ['random_forest', 'svm', 'gradient_boosting', 'mlp']
    elif args.model != 'cnn':
        traditional_models = [args.model]
    else:
        traditional_models = []
    
    for model_type in traditional_models:
        try:
            results = train_traditional_model(
                config, model_type, 
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"Error training {model_type}: {e}")
    
    # Save all results
    results_path = Path(config['paths']['results_dir']) / 'all_results.json'
    save_results(all_results, str(results_path))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {config['paths']['results_dir']}")
    print(f"Models saved to: {config['paths']['models_dir']}")


if __name__ == '__main__':
    main()
