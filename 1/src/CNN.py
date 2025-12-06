import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        # three convolutional layers with 32, 64, 128 channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # after 3 pooling layers: 28 -> 14 -> 7 -> 3
        # so: 128 channels × 3×3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # conv1 + Pool: [batch, 3, 28, 28] -> [batch, 32, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # conv2 + Pool: [batch, 32, 14, 14] -> [batch, 64, 7, 7]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # conv3 + Pool: [batch, 64, 7, 7] -> [batch, 128, 3, 3]
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # flatten: [batch, 128, 3, 3] -> [batch, 1152]
        x = x.view(x.size(0), -1)

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
