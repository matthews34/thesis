import torch
from torch import nn
import torch.nn.functional as F
import logging
from src import features, device
from src.utils.features import normalize, gen_PDP, rss, tof, power_first_path, delay_spread

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer
        input_dim = 0
        for f in features:
            if f == 'rss':
                input_dim += 64
            elif f == 'tof':
                input_dim += 64
            elif f == 'pofp':
                input_dim += 64
            elif f == 'ds':
                input_dim += 64
            elif f == 'csi':
                input_dim += 6400
            else:
                logging.error(f'Features: unknown feature {f}')
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

        output = x

        return output.float()
