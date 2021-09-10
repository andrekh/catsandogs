import torch.nn as nn
import torch
import numpy as np


# modelA_dim128.py
# works with dim = 128

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model_name = "3 layers Conv2d, 3 linear layers"
        self.dim = input_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(6272, 100)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.act5 = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = self.pool3(self.act3(self.conv3(out)))
        out = torch.flatten(out, start_dim=1)
        out = self.act4(self.fc1(out))
        out = self.fc3(self.act5(self.fc2(out)))
        return out

    def getTrainableParameters(self):
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_params])
        return trainable_params

    def getModelName(self):
        return self.model_name
