# Changelog

All notable changes to the Blood Discriminator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-20

### Added

#### Project Structure
- Complete repository structure with modular organization
- Configuration management with YAML files
- Comprehensive .gitignore for Python ML projects
- Package setup with setup.py
- Requirements.txt with all dependencies

#### Machine Learning Components
- Data preprocessing module with:
  - Image loading and normalization
  - Data augmentation (flip, rotation, brightness)
  - Color feature extraction (RGB, HSV, LAB)
  - Train/validation/test split functionality
  
- CNN classifier with:
  - 3-block convolutional architecture
  - Batch normalization and dropout
  - Configurable hyperparameters
  - Model saving/loading capabilities
  
- Traditional ML classifiers:
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  
- Comprehensive evaluation module with metrics:
  - Quality: Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa, AUC-ROC
  - Efficacy: Sensitivity, Specificity, Balanced Accuracy
  - Efficiency: Training time, Inference time, Throughput
  - Diversity: Prediction entropy, Class distribution, Confidence
  
- Utility functions for:
  - Configuration loading/saving
  - Results management
  - Directory creation

#### Training Pipeline
- Main training script (src/train.py) with:
  - Support for all model types
  - Configurable via YAML
  - Automatic evaluation and reporting
  - Model persistence
  - Command-line interface

#### Documentation
- Comprehensive README.md with:
  - Project overview
  - Installation instructions
  - Usage examples
  - Architecture description
  
- Springer LNCS format research report with:
  - Detailed architecture description
  - Complete experimental parameters table
  - Comprehensive evaluation metrics documentation
  - Results and discussion sections
  
- Quick Start Guide (QUICKSTART.md)
- Contributing Guidelines (CONTRIBUTING.md)
- Directory-specific README files
- LaTeX report compilation instructions

#### Testing
- Unit tests for all core components:
  - Data preprocessing
  - Model training and prediction
  - Evaluation metrics
  - Utility functions

#### Examples
- Jupyter notebook (01_exploratory_analysis.ipynb) with:
  - Data loading examples
  - Model training demonstrations
  - Visualization examples
  - Performance comparison

#### Configuration
- Experiment configuration (experiment_config.yaml) with:
  - Data parameters
  - Model architecture settings
  - Training hyperparameters
  - Traditional ML model parameters
  - File paths

### Project Features

- Multi-model comparison framework
- Reproducible experiments with fixed random seeds
- Comprehensive metrics covering quality, efficacy, efficiency, and diversity
- Automated report generation
- Confusion matrix visualization
- Support for both CNN and traditional ML approaches
- Modular and extensible architecture
- Full documentation for replication

### Dependencies

- TensorFlow/Keras >= 2.8.0
- scikit-learn >= 1.0.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- PyYAML >= 6.0
- pytest >= 7.0.0

## [Unreleased]

### Planned Features
- Support for Rh factor detection
- Real-time inference API
- Mobile deployment support
- Extended data augmentation techniques
- Transfer learning from pre-trained models
- Ensemble model combination
- Cross-validation support
- Hyperparameter optimization framework
