"""
Módulo de classificação supervisionada.
Contém o pipeline completo de treinamento e avaliação do modelo ANN.
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import random
from metrics import compute_metrics, print_metrics


# ---------------------------------------------------------
# Configurações
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE_EMB = 64
BATCH_SIZE_TRAIN = 64
EPOCHS = [5, 10, 20]
LR = 1e-3
SEED = 2025

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch):", torch.version.cuda)

if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Current GPU:", torch.cuda.current_device())
else:
    print("No GPU detected by PyTorch")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    """Define seeds para reprodutibilidade."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Modelo
# ---------------------------------------------------------

class ANNClassifier(nn.Module):
    """Rede neural artificial para classificação."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Funções de Pipeline
# ---------------------------------------------------------

def load_emotion_dataset():
    """Carrega o dataset de emoções."""
    print("Loading dataset...")
    ds = load_dataset("dair-ai/emotion")
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    label_names = train_ds.features["label"].names
    num_classes = len(label_names)

    X_train_texts = train_ds["text"]
    y_train = np.array(train_ds["label"])
    X_val_texts = val_ds["text"]
    y_val = np.array(val_ds["label"])
    X_test_texts = test_ds["text"]
    y_test = np.array(test_ds["label"])

    print(f"Labels: {label_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(X_train_texts)}")
    print(f"Val samples: {len(X_val_texts)}")
    print(f"Test samples: {len(X_test_texts)}")

    return (
        X_train_texts,
        y_train,
        X_val_texts,
        y_val,
        X_test_texts,
        y_test,
        label_names,
        num_classes,
    )


def load_embedding_model(model_name: str):
    """Carrega o modelo de embeddings."""
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def compute_embeddings(model, X_train_texts, X_val_texts, X_test_texts):
    """Gera embeddings para os textos."""
    print("\nComputing embeddings...")
    X_train = model.encode(
        X_train_texts,
        batch_size=BATCH_SIZE_EMB,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    X_val = model.encode(
        X_val_texts,
        batch_size=BATCH_SIZE_EMB,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    X_test = model.encode(
        X_test_texts,
        batch_size=BATCH_SIZE_EMB,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(f"\nEmbeddings shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    return X_train, X_val, X_test


def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    """Prepara DataLoaders do PyTorch."""
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def train_ann_classifier(epochs, input_dim, num_classes, train_loader, val_loader, device):
    """Treina o classificador ANN."""
    ann = ANNClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(ann.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining ANN classifier for {epochs} epochs...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        ann.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = ann(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        ann.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = ann(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                true.extend(yb.numpy().tolist())

        val_acc = accuracy_score(true, preds)
        print(f"  Epoch {epoch}/{epochs}: val_acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = ann.state_dict()

    # Carregar melhor modelo
    if best_state is not None:
        ann.load_state_dict(best_state)
        print(f"  Best val_acc: {best_val_acc:.4f}")

    return ann


def evaluate_on_test(model, test_loader, device):
    """Avalia o modelo no conjunto de teste."""
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            true.extend(yb.numpy().tolist())

    return true, preds


def train_and_evaluate_supervised():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    
    (
        X_train_texts,
        y_train,
        X_val_texts,
        y_val,
        X_test_texts,
        y_test,
        label_names,
        num_classes,
    ) = load_emotion_dataset()

    emb_model = load_embedding_model(MODEL_NAME)

    X_train, X_val, X_test = compute_embeddings(
        emb_model, X_train_texts, X_val_texts, X_test_texts
    )
    input_dim = X_train.shape[1]

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    results = {}
    
    for epochs in EPOCHS:
        print(f"# Training with {epochs} epochs")
        
        # Treinar modelo
        ann = train_ann_classifier(
            epochs, input_dim, num_classes, train_loader, val_loader, DEVICE
        )

        # Avaliar no conjunto de teste
        y_true, y_pred = evaluate_on_test(ann, test_loader, DEVICE)
        
        # Calcular métricas
        metrics = compute_metrics(y_true, y_pred, label_names)
        
        # Exibir resultados
        print_metrics(metrics, label_names, model_name=f"ANN Supervised ({epochs} epochs)")
        
        # Armazenar resultados
        results[f"{epochs}_epochs"] = {
            'model': ann,
            'metrics': metrics,
            'label_names': label_names
        }
    
    return results, (X_test_texts, y_test, label_names)
