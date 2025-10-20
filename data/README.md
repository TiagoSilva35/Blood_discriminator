# Data Directory

This directory contains all data-related files for the Blood Discriminator project.

## Structure

- **raw/**: Raw blood sample images organized by class
- **processed/**: Preprocessed and augmented data
- **results/**: Evaluation results, metrics, and visualizations

## Raw Data Format

Place your blood sample images in the `raw/` directory organized by blood type:

```
raw/
├── A/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
├── B/
│   ├── sample_001.jpg
│   └── ...
├── AB/
│   └── ...
└── O/
    └── ...
```

### Image Requirements

- **Format**: JPEG or PNG
- **Recommended size**: At least 224×224 pixels
- **Color space**: RGB
- **Quality**: High-resolution microscopy images preferred

## Processed Data

After running the preprocessing pipeline, processed data will be stored in `processed/`:

- Normalized images (224×224×3, float32, [0,1] range)
- Augmented training data
- Extracted features (optional)

## Results

After training and evaluation, results are saved in `results/`:

- `all_results.json`: Comprehensive metrics for all models
- `{model}_evaluation.txt`: Detailed text reports
- `{model}_confusion_matrix.png`: Confusion matrix visualizations
- Training history plots (if saved)

## Data Privacy

**Important**: Ensure that any real blood sample data complies with:
- Medical data privacy regulations (HIPAA, GDPR, etc.)
- Institutional review board (IRB) approvals
- Patient consent requirements

Do not commit sensitive medical data to version control.
