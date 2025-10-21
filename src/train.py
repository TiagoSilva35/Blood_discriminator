"""
Training loop for neural networks (both DNN and CNN)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score

def fit(device, X_train, y_train, nn, criterion, optimizer, n_epochs, to_device=True, batch_size=32):
    if to_device:
        nn = nn.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)

    # Train the network
    loss_values = []
    for epoch in range(n_epochs):
        accu_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            end_idx = min(i + batch_size, X_train.size(0))
            X_batch = X_train[i:end_idx]
            y_batch = y_train[i:end_idx]

            optimizer.zero_grad()
            # Forward pass
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_loss += loss.item()

            # Backward and optimize

            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, accu_loss))
        loss_values.append(accu_loss)

    return loss_values, nn.to("cpu")


def evaluate(device, X_test, y_test, nn,  to_device=True, batch_size=32):
    #send everything to the device (ideally a GPU)
    nn.eval()
    
    if to_device:
        nn = nn.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

    with torch.no_grad():
        outputs = nn(X_test)
        _, predicted = torch.max(outputs.data, 1)

    if to_device:
        predicted = predicted.to("cpu")
    predicted = predicted.numpy()
    y_test = y_test.to("cpu").numpy()
    CM = confusion_matrix(y_test, predicted)
    f1 = f1_score(y_test, predicted, average='weighted')

    return CM, f1

