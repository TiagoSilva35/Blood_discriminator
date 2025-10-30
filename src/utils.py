import os
import torch
import numpy as np

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