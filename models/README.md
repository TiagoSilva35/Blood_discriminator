# Models Directory

This directory stores trained machine learning models.

## Saved Models

After training, the following models are saved here:

- `cnn_model.h5`: Trained CNN model (Keras/TensorFlow format)
- `random_forest.pkl`: Random Forest model (scikit-learn pickle format)
- `svm.pkl`: SVM model (scikit-learn pickle format)
- `gradient_boosting.pkl`: Gradient Boosting model (scikit-learn pickle format)
- `mlp.pkl`: MLP model (scikit-learn pickle format)

## Loading Saved Models

### CNN Model

```python
from tensorflow import keras

model = keras.models.load_model('models/cnn_model.h5')
predictions = model.predict(X_test)
```

### Traditional ML Models

```python
import pickle

with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)
```

## Model Versioning

Consider using a model versioning system like:
- MLflow
- DVC (Data Version Control)
- Weights & Biases

For production deployments, track:
- Model version
- Training date
- Performance metrics
- Hyperparameters
- Training data version
