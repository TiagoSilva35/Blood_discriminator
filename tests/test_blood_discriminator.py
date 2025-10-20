"""
Unit tests for the blood discriminator system
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing.data_loader import BloodDataPreprocessor, split_data
from models.classifier import CNNBloodClassifier, TraditionalMLClassifier
from evaluation.metrics import ModelEvaluator


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = BloodDataPreprocessor(img_size=(224, 224), normalize=True)
        assert preprocessor.img_size == (224, 224)
        assert preprocessor.normalize == True
    
    def test_extract_color_features(self):
        """Test color feature extraction."""
        preprocessor = BloodDataPreprocessor()
        image = np.random.rand(224, 224, 3).astype(np.float32)
        features = preprocessor.extract_color_features(image)
        
        # Should extract 18 features (3 color spaces × 3 channels × 2 stats)
        assert features.shape == (18,)
    
    def test_augment_image(self):
        """Test image augmentation."""
        preprocessor = BloodDataPreprocessor(normalize=True)
        image = np.random.rand(224, 224, 3).astype(np.float32)
        augmented = preprocessor.augment_image(image)
        
        assert augmented.shape == image.shape
        assert augmented.dtype == np.float32


class TestDataSplitting:
    """Test data splitting functionality."""
    
    def test_split_data(self):
        """Test data splitting with correct ratios."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 4, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )
        
        assert len(X_train) == 70
        assert len(X_val) == 15
        assert len(X_test) == 15
        assert len(y_train) == 70
        assert len(y_val) == 15
        assert len(y_test) == 15


class TestCNNClassifier:
    """Test CNN classifier."""
    
    def test_model_initialization(self):
        """Test CNN model initializes correctly."""
        model = CNNBloodClassifier(
            input_shape=(224, 224, 3),
            num_classes=4,
            learning_rate=0.001
        )
        
        assert model.input_shape == (224, 224, 3)
        assert model.num_classes == 4
        assert model.learning_rate == 0.001
        assert model.model is not None
    
    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        model = CNNBloodClassifier(input_shape=(224, 224, 3), num_classes=4)
        X_test = np.random.rand(10, 224, 224, 3).astype(np.float32)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 4)


class TestTraditionalMLClassifier:
    """Test traditional ML classifiers."""
    
    def test_random_forest_initialization(self):
        """Test Random Forest initialization."""
        model = TraditionalMLClassifier(model_type='random_forest', n_estimators=10)
        assert model.model_type == 'random_forest'
    
    def test_svm_initialization(self):
        """Test SVM initialization."""
        model = TraditionalMLClassifier(model_type='svm')
        assert model.model_type == 'svm'
    
    def test_gradient_boosting_initialization(self):
        """Test Gradient Boosting initialization."""
        model = TraditionalMLClassifier(model_type='gradient_boosting')
        assert model.model_type == 'gradient_boosting'
    
    def test_mlp_initialization(self):
        """Test MLP initialization."""
        model = TraditionalMLClassifier(model_type='mlp')
        assert model.model_type == 'mlp'
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            TraditionalMLClassifier(model_type='invalid_model')
    
    def test_training_and_prediction(self):
        """Test training and prediction pipeline."""
        model = TraditionalMLClassifier(model_type='random_forest', n_estimators=10)
        
        X_train = np.random.rand(50, 224, 224, 3).astype(np.float32)
        y_train = np.random.randint(0, 4, 50)
        
        model.train(X_train, y_train)
        
        X_test = np.random.rand(10, 224, 224, 3).astype(np.float32)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 4)


class TestModelEvaluator:
    """Test model evaluation metrics."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = ModelEvaluator(class_names=['A', 'B', 'AB', 'O'])
        assert evaluator.class_names == ['A', 'B', 'AB', 'O']
    
    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # Perfect predictions
        
        metrics = evaluator.evaluate_quality(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect accuracy
    
    def test_efficacy_metrics(self):
        """Test efficacy metrics calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        
        metrics = evaluator.evaluate_efficacy(y_true, y_pred)
        
        assert 'mean_sensitivity' in metrics
        assert 'mean_specificity' in metrics
        assert 'balanced_accuracy' in metrics
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        evaluator = ModelEvaluator()
        
        metrics = evaluator.evaluate_efficiency(
            inference_time=1.5,
            n_samples=100,
            training_time=60.0
        )
        
        assert 'total_inference_time' in metrics
        assert 'inference_time_per_sample' in metrics
        assert 'throughput' in metrics
        assert 'total_training_time' in metrics
        assert metrics['inference_time_per_sample'] == 0.015
        assert metrics['throughput'] == pytest.approx(66.67, 0.01)
    
    def test_diversity_metrics(self):
        """Test diversity metrics calculation."""
        evaluator = ModelEvaluator()
        
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        
        metrics = evaluator.evaluate_diversity(y_pred)
        
        assert 'prediction_entropy' in metrics
        assert 'n_unique_predictions' in metrics
        assert metrics['n_unique_predictions'] == 4
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        
        results = evaluator.comprehensive_evaluation(
            y_true, y_pred,
            inference_time=1.0,
            n_samples=8,
            training_time=30.0
        )
        
        assert 'quality' in results
        assert 'efficacy' in results
        assert 'efficiency' in results
        assert 'diversity' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
