import os
import torch
import numpy as np

def write_log(file_path, content):
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    file_path = 'logs/' + file_path
    with open(file_path, 'w') as file:
        file.write(content)
    
def get_device():
    device = None
    if torch.backends.mps.is_available():
       device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    return device

def process_data(data_loader, flag = True):
    if flag:
        X, y = [], []
        
        for batch_X, batch_y in data_loader:
            X.append(batch_X)
            y.append(batch_y.squeeze())  # Remove extra dimensions
        
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        return X.reshape(X.shape[0], -1), y 
    else:
        X, y = [], []
        
        for batch_X, batch_y in data_loader:
            X.append(batch_X)
            y.append(batch_y.squeeze())
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        write_log('data_shapes.txt', f'X shape: {X.shape}\ny shape: {y.shape}\n')
        return X, y