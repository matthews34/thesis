import torch
from torch import nn
import torch.nn.functional as F
import logging

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(200, 16000),
            nn.ReLU(),
            nn.Linear(16000, 8000),
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

    def prepare_input(self, x):
        CFR = x                         # CFR (CSI) - shape = (B, 64, 100)
        CIR = torch.fft.ifft(x, dim=2)  # CIR - shape = (B, 64, 100)

        # extract RSS
        RSS = torch.sum(torch.square(torch.abs(CFR)), dim=1) # sum over antennas, shape should be (B, 100)

        # extract ToF
        pdp = torch.square(torch.abs(CIR)) # sum over antennas, shape should be (B, 64, 100)
        ToF = torch.argmax(pdp, dim=1) # shape should be (B, 100)

        # concatenate features
        x = torch.stack((RSS, ToF), dim=-1)
        x = torch.flatten(x, start_dim=1) # shape = (B, 200)

        return x.float()
