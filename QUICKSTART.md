# Quick Start Guide - Blood Discriminator

This guide will help you get started with the Blood Discriminator system quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster CNN training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TiagoSilva35/Blood_discriminator.git
cd Blood_discriminator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import tensorflow; import sklearn; print('Installation successful!')"
```

## Quick Training Example

### Using Synthetic Data (Demo)

The system includes a demo mode with synthetic data:

```bash
python src/train.py --config config/experiment_config.yaml
```

This will:
- Generate synthetic blood sample data
- Train all models (CNN, Random Forest, SVM, Gradient Boosting, MLP)
- Evaluate performance with comprehensive metrics
- Save results to `data/results/`
- Save trained models to `models/`

### Training Specific Models

Train only the CNN:
```bash
python src/train.py --model cnn
```

Train only Random Forest:
```bash
python src/train.py --model random_forest
```

## Using Your Own Data

1. **Organize your data** in the following structure:
```
data/raw/
├── A/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
├── B/
│   └── ...
├── AB/
│   └── ...
└── O/
    └── ...
```

2. **Update the training script** to load real images instead of synthetic data:

Edit `src/train.py` and replace the synthetic data generation section with:

```python
from preprocessing.data_loader import BloodDataPreprocessor

preprocessor = BloodDataPreprocessor(img_size=(224, 224), normalize=True)
X, y = preprocessor.preprocess_dataset(
    data_dir=config['paths']['data_dir'],
    labels_file='path/to/labels.csv'  # Optional
)
```

3. **Train the models**:
```bash
python src/train.py --data-dir data/raw
```

## Viewing Results

After training, check the results:

```bash
# View comprehensive metrics
cat data/results/cnn_evaluation.txt

# View all results
cat data/results/all_results.json

# Open confusion matrices
# Located in: data/results/*_confusion_matrix.png
```

## Using Jupyter Notebooks

For interactive exploration:

```bash
jupyter notebook
# Navigate to notebooks/01_exploratory_analysis.ipynb
```

## Configuration

Edit `config/experiment_config.yaml` to customize:

```yaml
# Change image size
model:
  input_shape: [224, 224, 3]  # Modify as needed

# Adjust training parameters
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

# Modify data split
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## Common Tasks

### Change Number of Epochs

Edit `config/experiment_config.yaml`:
```yaml
training:
  epochs: 100  # Increase for more training
```

### Adjust Learning Rate

Edit `config/experiment_config.yaml`:
```yaml
training:
  learning_rate: 0.0001  # Lower for finer optimization
```

### Use Different Image Size

Edit `config/experiment_config.yaml`:
```yaml
model:
  input_shape: [128, 128, 3]  # Smaller for faster training

preprocessing:
  img_size: [128, 128]
```

### Add Data Augmentation

Data augmentation is already implemented in `BloodDataPreprocessor.augment_image()`. Enable it during training by calling:

```python
augmented_img = preprocessor.augment_image(image)
```

## Model Evaluation Metrics

The system evaluates models across four dimensions:

1. **Quality**: Accuracy, Precision, Recall, F1-score
2. **Efficacy**: Sensitivity, Specificity
3. **Efficiency**: Training time, Inference speed
4. **Diversity**: Prediction confidence, Class distribution

All metrics are automatically computed and saved.

## Troubleshooting

### Out of Memory Errors

Reduce batch size in `config/experiment_config.yaml`:
```yaml
training:
  batch_size: 16  # Smaller batch size
```

### Slow Training

- Use GPU acceleration (CUDA)
- Reduce image size
- Use fewer epochs
- Train traditional ML models only (faster than CNN)

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Prepare your data**: Organize blood sample images
2. **Run initial training**: Use demo mode to verify setup
3. **Train on real data**: Replace synthetic data with actual images
4. **Optimize hyperparameters**: Experiment with different settings
5. **Generate report**: Compile the LaTeX report with your results

## Getting Help

- Check `README.md` for detailed documentation
- Review `docs/report/README.md` for report compilation
- Examine `notebooks/01_exploratory_analysis.ipynb` for examples
- Read code comments in `src/` modules

## Citation

If you use this system in your research, please cite:
```bibtex
@software{blood_discriminator,
  title={Blood Discriminator - Machine Learning System},
  author={Author Name},
  year={2024},
  url={https://github.com/TiagoSilva35/Blood_discriminator}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: your.email@example.com
