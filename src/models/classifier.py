"""
Machine Learning Models for Blood Type Classification
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class CNNBloodClassifier:
    """
    Convolutional Neural Network for blood type classification.
    Architecture: Conv blocks + Dense layers
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                 num_classes: int = 4,
                 learning_rate: float = 0.001):
        """
        Initialize CNN classifier.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of blood type classes
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        Build CNN architecture.
        
        Returns:
            Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNNBloodClassifier')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50, 
              batch_size: int = 32) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input images
            
        Returns:
            Predicted class labels
        """
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)


class TraditionalMLClassifier:
    """
    Traditional ML classifier wrapper for blood type classification.
    Supports Random Forest, SVM, Gradient Boosting, and MLP.
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'gradient_boosting', 'mlp')
            **kwargs: Additional parameters for the specific classifier
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        
    def _create_model(self, **kwargs):
        """Create the specified model type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (256, 128)),
                activation=kwargs.get('activation', 'relu'),
                learning_rate_init=kwargs.get('learning_rate_init', 0.001),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Flatten images if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For SVM, use decision_function
            decisions = self.model.decision_function(X)
            # Convert to probabilities using softmax
            exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
            return exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)
