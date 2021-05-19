# %%
import torch
from torch import nn
import torch.nn.functional as F
import logging

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(12800, 8000),
            nn.ReLU(),
            nn.Linear(8000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            
            nn.Linear(100, 2)
        )
        
    def forward(self, x):
        x = self.prepare_input(x)
        x = self.model(x)
        return x

    # Vectorize samples
    # TODO: flexibilize size ?
    def prepare_input(self, x):
        # concatenate into one 6400x1 vector
        x = torch.flatten(x, start_dim=1)
        # split complex values
        x = torch.stack((x.real, x.imag), -1) # vector is now 6400x2
        # concatenate into one 12800x1 vector
        x = torch.flatten(x, start_dim=1)
        return x.float()
