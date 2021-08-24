import torch
from torch import nn
import torch.nn.functional as F
import logging
from src import device, scenario

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer
        if scenario == 'subcarriers1':
            input_dim = 128
        elif scenario == 'subcarriers5':
            input_dim = 640
        elif scenario == 'subcarriers10':
            input_dim = 1280
        elif scenario == 'subcarriers25':
            input_dim = 3200
        elif scenario == 'subcarriers50':
            input_dim = 6400
        elif scenario == 'subcarriers75':
            input_dim = 9600
        else:
            logging.error(f'Subcarriers: invalid scenario {scenario}')
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
        if scenario == 'subcarriers25':
            indices = torch.tensor([4*i for i in range(25)]).to(device)
        elif scenario == 'subcarriers50':
            indices = torch.tensor([2*i for i in range(50)]).to(device)
        elif scenario == 'subcarriers75':
            indices = torch.tensor(
                list(
                    set([i for i in range(100)])-set([4*i for i in range(25)])
                )
            ).to(device)
        elif scenario == 'subcarriers10':
            indices = torch.tensor([10*i for i in range(10)]).to(device)
        elif scenario == 'subcarriers5':
            indices = torch.tensor([20*i for i in range(5)]).to(device)
        elif scenario == 'subcarriers1':
            indices = torch.tensor([0]).to(device)
        else:
            logging.error(f'Subcarriers: invalid scenario {scenario}')
            exit(-1)
        
        # select indices
        x = torch.index_select(x, 2, indices)
        # concatenate into one vector
        x = torch.flatten(x, start_dim=1)
        # split complex values
        x = torch.stack((x.real, x.imag), -1)
        # concatenate into one vector
        x = torch.flatten(x, start_dim=1)

        return x.float()
