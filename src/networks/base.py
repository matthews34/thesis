import torch
from torch import nn
import torch.nn.functional as F
import logging

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 4, kernel_size=9, padding=4) # 25600
        # pooling(1,5) -> 5120
        self.conv2 = nn.Conv2d(4, 8, kernel_size=9, padding=4) # 10240
        # pooling(1,5) -> 2018
        self.conv3 = nn.Conv2d(8, 16, kernel_size=9, padding=4) # 4096
        self.conv4 = nn.Conv2d(16, 4, kernel_size=9, padding=4) # 1024
        # pooling(1,2) -> 512
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)

        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.prepare_input(x)
        x = self.cnn(x)
        x = self.fc(x)
        x = self.output(x)
        return x

    def prepare_input(self, x):
        # x.shape = (N x 64 x 100)
        real =  x.real
        imag = x.imag
        rectangular = torch.stack((real, imag), 1)
        
        # input.shape = (N x 2 x 64 x 100)
        input = rectangular

        return input.float()

    def cnn(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(x, kernel_size=(1,5))
        x = self.conv2(x)
        x = F.avg_pool2d(x, kernel_size=(1,5))
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.avg_pool2d(x, kernel_size=(1,2))

        return torch.flatten(x, 1)
        
    def fc(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x