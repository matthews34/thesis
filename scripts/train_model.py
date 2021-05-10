# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import necessary packages

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import sys

parent_dir = Path(Path.absolute(Path(__file__))).parent.parent
sys.path.insert(0, str(parent_dir))
from utils import load_posistions, load_data, timed_print, running_from_ipython

CONFIG_FILE = parent_dir.joinpath('.config_ipynb')

# %%
# Receive parameters as arguments
import argparse

if running_from_ipython():
    with open(CONFIG_FILE) as config:
        sys.argv = config.read().split()

parser = argparse.ArgumentParser(description='Train the neural network model')
parser.add_argument('network_module', type=str, help='Path to the module containing the network')
parser.add_argument('dataset_dir', type=str, help='Path to dataset root directory')
parser.add_argument('epochs', type=int)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--num_workers', type=int, default=8)

# Set parameters
args = parser.parse_args()
network_module = args.network_module
dataset_dir = Path(args.dataset_dir)
epochs = args.epochs
batch_size = args.batch_size
num_workers = args.num_workers


# %%
# Configure dataset
class CSIDataset(Dataset):
    """CSI dataset."""
    
    def __init__(self, positions_file, samples_dir, indices_file):
        """
        Args:
            positions_file (string): Path to the file containing the user positions.
            samples_dir (string): Directory containing the samples.
            indexes_file (string): Path to the file holding the indexes to be considered for the set
        """
        self.user_positions = load_data(positions_file)
        self.samples_dir = samples_dir
        self.indices = load_data(indices_file)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        
        sample_filepath = Path.joinpath(self.samples_dir, 'channel_measurement_{:06d}.npy'.format(index))
        sample = load_data(sample_filepath)
                    
        # Remove z coordinate from the positions
        label = np.delete(self.user_positions[index], -1)
        
        return sample, label

# %%
# Create dataset and dataloader objects

positions_file = dataset_dir.joinpath('user_positions.npy')
samples_dir = dataset_dir.joinpath('samples')
indices_file = dataset_dir.joinpath('train_indices.npy')

train_dataset = CSIDataset(positions_file, samples_dir, indices_file)
test_dataset = CSIDataset(positions_file, samples_dir, indices_file)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# %%
# Import network and vectorization function
from importlib import import_module
network = import_module(network_module[:-3])
Network = network.Network
# vectorize_samples = network.vectorize_samples
model = Network()
timed_print(f'Network created from {network_module}')

# %%
from torch import optim

# Choose GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
model.to(device)

timed_print('Running on ' + ('GPU' if device.type == 'cuda' else 'CPU'))

# %%
# TODO: capture keyboard interrupt
import torch.nn as nn
# Define the loss as MSE loss
criterion = nn.MSELoss()
# Define optimizer to update weights (stochastic gradient descent)
optimizer = optim.SGD(model.parameters(), lr=0.001)

timed_print('Starting training...')

training_losses = []
test_losses = []
for e in range(epochs):
    training_loss = 0
    testing_loss = 0
    for samples, labels in train_loader:
        
        samples, labels = samples.to(device), labels.float().to(device)
        
        # Vectorize the samples
        # samples = vectorize_samples(samples)
    
        optimizer.zero_grad()
        
        output = model(samples)
        loss = criterion(output, labels)
        loss.backward()
        
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        training_loss += loss.item()
        #print(loss.item())
    else:
        with torch.no_grad():
            for samples, labels in test_loader:
                samples, labels = samples.to(device), labels.to(device)
        
                # Vectorize the samples
                samples = vectorize_samples(samples)

                output = model(samples.float())
                loss = criterion(output, labels.float())

                testing_loss += loss.item()
                
        training_loss /= len(train_dataset)
        testing_loss /= len(test_dataset)
        training_losses.append(training_loss)
        test_losses.append(testing_loss)

        timed_print(
            "Epoch: {}/{}... ".format(e+1, epochs),
            "Training Loss: {:.3f}... ".format(training_loss),
            "Test Loss: {:.3f}... ".format(testing_loss)
        )
        

# %%
# save model and losses
from pathlib import Path

basename = Path(network_module).name
losses_path = Path.joinpath(parent_dir, 'output', 'losses', basename[:-3] + '_losses.npy')
model_path = Path.joinpath(parent_dir, 'output', 'models', basename[:-3] + '_model')
Path(losses_path.parent).mkdir(parents=True, exist_ok=True)
Path(model_path.parent).mkdir(parents=True, exist_ok=True)
np.save(losses_path, (training_losses, test_losses))
torch.save(model.state_dict(), model_path)