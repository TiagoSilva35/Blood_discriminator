import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        """
        Multi-Layer Perceptron (MLP) with configurable hidden layers and dropout.
        """
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        
        return x
