"""
Data preprocessing module for blood sample images/data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import cv2
from pathlib import Path


class BloodDataPreprocessor:
    """
    Preprocessor for blood sample data.
    Handles image loading, normalization, augmentation, and feature extraction.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target image size (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.img_size = img_size
        self.normalize = normalize
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        if self.normalize:
            img = img.astype(np.float32) / 255.0
            
        return img
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to an image.
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image array
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1 if self.normalize else 255)
        
        return image
    
    def preprocess_dataset(self, data_dir: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess entire dataset.
        
        Args:
            data_dir: Directory containing images
            labels_file: Optional CSV file with labels
            
        Returns:
            Tuple of (images, labels) arrays
        """
        data_path = Path(data_dir)
        image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
        
        images = []
        for img_file in image_files:
            img = self.load_image(str(img_file))
            images.append(img)
            
        images = np.array(images)
        
        labels = None
        if labels_file:
            df = pd.read_csv(labels_file)
            labels = df['label'].values
            
        return images, labels
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color-based features from blood sample image.
        
        Args:
            image: Input image array
            
        Returns:
            Feature vector
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Extract mean and std for each channel
        features = []
        for img_space in [image, hsv, lab]:
            for channel in range(3):
                features.append(np.mean(img_space[:, :, channel]))
                features.append(np.std(img_space[:, :, channel]))
                
        return np.array(features)


def split_data(X: np.ndarray, y: np.ndarray, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features array
        y: Labels array
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
