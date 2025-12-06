import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.activations = nn.ModuleList()
        
        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.activations.append(nn.ReLU())
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.activations.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    def forward(self, x):
        # flatten the input 
        # (batch_size, channels, height, width) -> (batch_size, input_size)
        x = x.view(x.size(0), -1)
        
        # pass through all hidden layers with activations
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activations[i](x)
            x = self.dropout(x)

        # pass through the final layer (no activation)
        x = self.layers[-1](x)
        
        return x
