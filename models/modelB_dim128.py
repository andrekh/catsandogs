import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# modelB_dim128.py
# works with dim = 128

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model_name = "2 layers Conv2d, maxpool, 2 layers conv2d, maxpool, 3 linear layers"
        self.dim = input_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)

        self.fc1 = nn.Linear(3136, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv2(self.conv1(x))), kernel_size=3)
        out = F.max_pool2d(F.relu(self.conv4(self.conv3(out))), kernel_size=5)
        out = torch.flatten(out, start_dim=1)
        out = self.fc3(F.relu(self.fc2(F.relu(self.fc1(out)))))
        return out

    def getTrainableParameters(self):
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_params])
        return trainable_params

    def getModelName(self):
        return self.model_name
