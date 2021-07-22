import torch
from torch import nn
import torch.nn.functional as F
import logging
from src import device, scenario

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer
        if scenario == 'antennas16':
            input_dim = 3200
        elif scenario == 'antennas32':
            input_dim = 6400
        elif scenario == 'antennas48':
            input_dim = 9600
        else:
            logging.error(f'Antennas: invalid scenario {scenario}')
            exit(-1)
        self.fc1 = nn.Linear(input_dim, 16000)
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(4,2), stride=(4,2)) 
        self.conv3 = nn.Conv2d(8, 20, kernel_size=(1,5), stride=(1,5)) 
        self.fc2 = nn.Linear(500, 2)
        
    def forward(self, x):

        x = self.prepare_input(x)
        
        # First layer (fully connected)
        x = self.fc1(x) # shape: 16000
        x = F.relu(x)
        
        # Reshape x to pass it through convolutional layers
        x = x.reshape((-1, 1, 80, 200)) # shape: 1 x 80 x 200

        x = self.conv1(x) # shape: 4 x 20 x 50 = 4000
        x = F.relu(x) # 4 x 20 x 50 = 4000
        
        x = self.conv2(x) # shape: 8 x 5 x 25 = 
        x = F.relu(x) # 8 x 10 x 25
        
        x = self.conv3(x) # 20 x 5 x 5
        x = F.relu(x) # 20 x 5 x 5 = 500
        
        # Flatten to run through last layer
        x = torch.flatten(x, 1) # shape: 500
        
        # Output layer
        x = self.fc2(x)
        
        return x

    def prepare_input(self, x):
        if scenario == 'antennas16':
            indices = torch.tensor(
                [3, 4, 11, 12, 19, 20, 27, 28,
                35, 36, 43, 44, 51, 52, 59, 60]
            ).to(device)
        elif scenario == 'antennas32':
            indices = torch.tensor(
                [2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21,
                26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44,
                45, 50, 51, 52, 53, 58, 59, 60, 61]
            ).to(device)
        elif scenario == 'antennas48':
            indices = torch.tensor(
                [i+n for i in range(1,60,8) for n in range(6)]
            ).to(device)
        else:
            logging.error(f'Antennas: invalid scenario {scenario}')
            exit(-1)
        
        # select indices
        x = torch.index_select(x, dim=1, indices)
        x = torch.flatten(x, start_dim=1)
        # split complex values
        x = torch.stack((x.real, x.imag), -1) # vector is now 3200x2
        # concatenate into one 6400x1 vector
        x = torch.flatten(x, start_dim=1)

        return x.float()