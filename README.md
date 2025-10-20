# Blood Discriminator - Machine Learning System

A comprehensive machine learning system for automated blood type classification using deep learning and traditional ML approaches.

## Overview

This project implements a complete pipeline for blood type discrimination, comparing multiple machine learning approaches:
- Convolutional Neural Networks (CNN)
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting
- Multi-Layer Perceptron (MLP)

The system classifies blood samples into four major ABO blood types: A, B, AB, and O.

## Project Structure

```
Blood_discriminator/
├── config/                      # Configuration files
│   └── experiment_config.yaml   # Experiment parameters
├── data/                        # Data directory
│   ├── raw/                     # Raw blood sample images
│   ├── processed/               # Processed data
│   └── results/                 # Evaluation results
├── docs/                        # Documentation
│   └── report/                  # Springer LNCS format report
│       ├── blood_discriminator_report.tex
│       └── README.md
├── models/                      # Saved trained models
├── notebooks/                   # Jupyter notebooks for exploration
├── src/                         # Source code
│   ├── preprocessing/           # Data preprocessing modules
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/                  # ML model implementations
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── evaluation/              # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── __init__.py
│   └── train.py                 # Main training script
├── tests/                       # Unit tests
├── .gitignore                   # Git ignore file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TiagoSilva35/Blood_discriminator.git
cd Blood_discriminator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training Models

Train all models with default configuration:
```bash
python src/train.py --config config/experiment_config.yaml
```

Train specific model:
```bash
# Train only CNN
python src/train.py --model cnn

# Train only Random Forest
python src/train.py --model random_forest

# Train only SVM
python src/train.py --model svm
```

### Configuration

Edit `config/experiment_config.yaml` to customize:
- Data split ratios
- Model hyperparameters
- Training parameters
- File paths

### Data Preparation

Place your blood sample images in the following structure:
```
data/raw/
├── A/
│   ├── sample_001.jpg
│   └── ...
├── B/
│   ├── sample_001.jpg
│   └── ...
├── AB/
│   └── ...
└── O/
    └── ...
```

## Architecture

### CNN Architecture

The Convolutional Neural Network consists of:
- 3 convolutional blocks (32, 64, 128 filters)
- Batch normalization and dropout for regularization
- Dense layers (256, 128 neurons)
- Softmax output layer (4 classes)

### Traditional ML Models

- **Random Forest**: Ensemble of 100 decision trees
- **SVM**: RBF kernel with optimized C and gamma parameters
- **Gradient Boosting**: 100 estimators with learning rate 0.1
- **MLP**: Two hidden layers (256, 128 neurons)

## Evaluation Metrics

The system provides comprehensive evaluation across four dimensions:

### 1. Quality Metrics
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-Score (macro and weighted)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- AUC-ROC

### 2. Efficacy Metrics
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Balanced Accuracy

### 3. Efficiency Metrics
- Training time
- Inference time
- Throughput (samples/second)

### 4. Diversity Metrics
- Prediction entropy
- Class distribution
- Confidence statistics

## Results

Results are automatically saved to `data/results/`:
- `all_results.json` - Comprehensive metrics for all models
- `{model}_evaluation.txt` - Detailed evaluation report per model
- `{model}_confusion_matrix.png` - Confusion matrix visualization

## Documentation

### Research Report

A comprehensive research report in Springer LNCS format is available in `docs/report/`. The report includes:

- General architecture description
- Detailed experimental setup with full replication parameters
- Comprehensive evaluation metrics
- Results and discussion

To compile the report:
```bash
cd docs/report
pdflatex blood_discriminator_report.tex
```

See `docs/report/README.md` for more details.

## Experimental Parameters

All experimental parameters are documented in:
1. `config/experiment_config.yaml` - Configuration file
2. `docs/report/blood_discriminator_report.tex` - Research report (Table 1)

Key parameters for replication:
- Image size: 224×224×3
- Train/Val/Test split: 70%/15%/15%
- CNN learning rate: 0.001
- Batch size: 32
- Epochs: 50 (with early stopping)

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Models

1. Implement the model in `src/models/`
2. Add configuration in `config/experiment_config.yaml`
3. Update `src/train.py` to include the new model

## Citation

If you use this code in your research, please cite:

```bibtex
@article{blood_discriminator,
  title={Blood Type Discrimination Using Machine Learning: A Comparative Study},
  author={Author Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- Author: [Your Name]
- Email: your.email@example.com
- GitHub: https://github.com/TiagoSilva35/Blood_discriminator

## Acknowledgments

[Add acknowledgments if applicable]