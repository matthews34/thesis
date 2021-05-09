# %%
import torch

# %%
# TODO: flexibilize size ?

# Vectorize samples
def vectorize_samples(samples):
    # concatenate into one 6400x1 vector
    samples = torch.flatten(samples, start_dim=1)
    # split complex values
    samples = torch.stack((samples.real, samples.imag), -1) # vector is now 6400x2
    # concatenate into one 12800x1 vector
    samples = torch.flatten(samples, start_dim=1)
    return samples

# %%
from torch import nn
import torch.nn.functional as F

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
        out = self.model(x)
        return out

