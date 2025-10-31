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


def fit(
    device,
    X_train,
    y_train,
    nn,
    criterion,
    optimizer,
    n_epochs,
    to_device=True,
    batch_size=32,
):
    # send everything to the device (ideally a GPU)
    if to_device:
        nn = nn.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)

    # train the network
    loss_values = []
    accuracy_values = []
    for epoch in range(n_epochs):
        accu_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        for i in range(0, X_train.size(0), batch_size):
            # select a batch (32 samples)
            end_idx = min(i + batch_size, X_train.size(0))
            X_batch = X_train[i:end_idx]
            y_batch = y_train[i:end_idx]
            
            # batch normalization
            optimizer.zero_grad()

            # forward pass
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_loss += loss.item()
            num_batches += 1
            
            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # backpropagation with optimization
            loss.backward()
            optimizer.step()

        avg_loss = accu_loss / num_batches
        accuracy = 100 * correct / total
        accuracy_values.append(accuracy)
        loss_values.append(avg_loss)
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, n_epochs, avg_loss))
        print("Training Accuracy: {:.2f}%".format(accuracy))
    return loss_values, accuracy_values, nn.to("cpu")


def evaluate(device, X_test, y_test, nn, criterion, to_device=True):
    assert criterion is not None, "Criterion must be provided for evaluation."
    # switch to evaluation mode
    nn.eval()

    # send everything to the device (ideally a GPU)
    if to_device:
        nn = nn.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

    # evaluate the model
    with torch.no_grad():
        outputs = nn(X_test)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, y_test)
        accuracy = 100 * (predicted == y_test).sum().item() / y_test.size(0)

    if to_device:
        predicted = predicted.to("cpu")

    predicted = predicted.numpy()
    y_test = y_test.to("cpu").numpy()

    # compute metrics
    CM = confusion_matrix(y_test, predicted)
    f1 = f1_score(y_test, predicted, average="weighted")

    return CM, f1, loss.item(), accuracy
