# Springer LNCS Report

This directory contains the research report in Springer LNCS (Lecture Notes in Computer Science) format.

## Files

- `blood_discriminator_report.tex` - Main LaTeX report file

## Compiling the Report

To compile the report, you need a LaTeX distribution installed (e.g., TeX Live, MiKTeX).

### Using pdflatex:

```bash
cd docs/report
pdflatex blood_discriminator_report.tex
bibtex blood_discriminator_report
pdflatex blood_discriminator_report.tex
pdflatex blood_discriminator_report.tex
```

### Using latexmk (recommended):

```bash
cd docs/report
latexmk -pdf blood_discriminator_report.tex
```

## Report Contents

The report includes:

1. **General Architecture Description**
   - System overview
   - CNN architecture details
   - Traditional ML models description
   - Data preprocessing pipeline

2. **Experimental Setup**
   - Complete parameter table for replication
   - Dataset description and splitting strategy
   - Hardware and software environment

3. **Evaluation Metrics**
   - **Quality metrics**: Accuracy, Precision, Recall, F1-score, MCC, Cohen's Kappa, AUC-ROC
   - **Efficacy metrics**: Sensitivity, Specificity, Balanced Accuracy
   - **Efficiency metrics**: Training time, Inference time, Throughput
   - **Diversity metrics**: Prediction entropy, Class distribution, Confidence statistics

4. **Results and Discussion**
   - Model performance comparison
   - Confusion matrices
   - Computational efficiency analysis

## LNCS Format

This report follows the Springer LNCS format guidelines:
- Document class: `llncs`
- Standard LNCS sections and formatting
- Proper bibliography style
- Tables and figures following LNCS guidelines
