import torch
from torch import nn
import torch.nn.functional as F
import logging

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer
        self.fc1 = nn.Linear(12800, 16000)
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=9, padding=4) 
        self.conv3 = nn.Conv2d(8, 20, kernel_size=9, padding=4) 
        self.fc2 = nn.Linear(500, 2)
        
    def forward(self, x):

        x = self.prepare_input(x)
        
        # First layer (fully connected)
        x = self.fc1(x) # shape: 16000
        x = F.relu(x)
        
        # Reshape x to pass it through convolutional layers
        x = x.reshape((-1, 1, 80, 200)) # shape: 1 x 80 x 200

        x = self.conv1(x) # shape: 4 x 80 x 200
        x = F.max_pool2d(x, 4)
        x = F.relu(x) # 4 x 20 x 50 = 4000
        
        x = self.conv2(x) # shape: 8 x 20 x 50
        x = F.max_pool2d(x, (4,2))
        x = F.relu(x) # 8 x 5 x 25 = 1000
        
        x = self.conv3(x) # 20 x 5 x 25
        x = F.max_pool2d(x, (1,5))
        x = F.relu(x) # 20 x 5 x 5 = 500
        
        # Flatten to run through last layer
        x = torch.flatten(x, 1) # shape: 500
        
        # Output layer
        x = self.fc2(x)
        
        return x

    def prepare_input(self, x):
        # concatenate into one 6400x1 vector
        x = torch.flatten(x, start_dim=1)
        # split complex values
        x = torch.stack((x.real, x.imag), -1) # vector is now 6400x2
        # concatenate into one 12800x1 vector
        x = torch.flatten(x, start_dim=1)

        return x.float()