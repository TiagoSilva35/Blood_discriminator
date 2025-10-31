import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def write_log(file_path, content):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    file_path = 'logs/' + file_path
    with open(file_path, 'w') as file:
        file.write(content)
    
def get_device():
    device = None
    # choose the correct device for training (optimization)
    if torch.backends.mps.is_available():
       device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    return device

def process_data(data_loader, flag = True):
    X, y = [], []
    for batch_X, batch_y in data_loader:
        X.append(batch_X)
        y.append(batch_y.squeeze())
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    if flag:        
        return X.reshape(X.shape[0], -1), y 
    else:
        write_log('data_shapes.txt', f'X shape: {X.shape}\ny shape: {y.shape}\n')
        return X, y
    
def get_metrics(cm):
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision = np.divide(TP, TP + FP, out = np.zeros_like(TP, dtype=float), where=(TP + FP)!=0)
    sensitivity = np.divide(TP, TP + FN, out = np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
    specificity = np.divide(TN, TN + FP, out = np.zeros_like(TP, dtype=float), where=(TN + FP)!=0)

    write_log('metrics.txt', f'Precision: {precision}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\n') 
    return precision, sensitivity, specificity

def plot_confusion_matrix(CM, n_classes):
    # Plot the CM
    plt.figure(figsize=(8, 6))
    sns.heatmap(CM, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(n_classes)],
                yticklabels=[str(i) for i in range(n_classes)]) 
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_acc(training_acc, test_acc):
    plt.figure()
    plt.plot(training_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()

def plot_loss(training_loss, test_loss):
    plt.figure()
    plt.plot(training_loss, label='Training Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()

def plot_precision_recall(precision, recall, n_classes):
    plt.figure(figsize=(10, 6))
    x = np.arange(n_classes)
    width = 0.35
    
    plt.bar(x - width/2, precision, width, label='Precision', alpha=0.8)
    plt.bar(x + width/2, recall, width, label='Recall (Sensitivity)', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision and Recall per Class')
    plt.xticks(x, [f'Class {i}' for i in range(n_classes)], rotation=45)
    plt.ylim([0, 1.0])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

def plot_complexity_vs_performance(complexity, performance):
    plt.figure()
    plt.plot(complexity, performance, marker='o')
    plt.xlabel('Model Complexity')
    plt.ylabel('Performance Metric')
    plt.title('Model Complexity vs Performance')
    plt.grid()
    plt.tight_layout()

def create_plots(training_acc, test_acc, training_loss, test_loss, precision, recall, n_classes):
    # plot_complexity_vs_performance(complexity, performance)
    plot_acc(training_acc, test_acc)
    plot_loss(training_loss, test_loss)
    plot_precision_recall(precision, recall, n_classes)
    plt.show()
