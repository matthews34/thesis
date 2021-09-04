import os
import random
import numpy as np
import torch
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from src.utils.data_manager import load_data
from src import NUM_SAMPLES, dataset_dir, batch_size, num_workers, training_size

class FeaturesRawDataset(Dataset):
    def __init__(self, positions_file, csi_dir, features_dir, indices):
        self.user_positions = load_data(positions_file)
        self.csi_dir = csi_dir
        self.features_dir = features_dir
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        csi_filepath = os.path.join(self.csi_dir, '{:06d}.pt'.format(index))
        csi = torch.load(csi_filepath)

        features_filepath = os.path.join(self.featuress_dir, '{:06d}.pt'.format(index))
        features = torch.load(features_filepath)

        # Remove z coordinate from the positions
        label = np.delete(self.user_positions[index], -1)

        shape = [1]
        shape.extend(csi.shape)
        csi = process_csi(torch.from_numpy(csi).reshape(shape))
        sample = torch.cat((csi, features), dim=-1)
                    
        return sample, label

def create_dataloader():
    positions_file = os.path.join(dataset_dir, 'user_positions.npy')
    samples_dir = os.path.join('data', 'features')
    training_indices, validation_indices, test_indices = generate_indices()

    # Setup dataset and data loader
    training_set = FeaturesDataset(positions_file, samples_dir, training_indices)
    validation_set = FeaturesDataset(positions_file, samples_dir, validation_indices)
    test_set = FeaturesDataset(positions_file, samples_dir, test_indices)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    training = SimpleNamespace(**{'loader': training_loader, 'size': len(training_set)})
    validation = SimpleNamespace(**{'loader': validation_loader, 'size': len(validation_set)})
    test = SimpleNamespace(**{'loader': test_loader, 'size': len(test_set)})

    return training, validation, test

def generate_indices():
    indices = [x for x in range(NUM_SAMPLES)]

    train_set_size = int(0.8 * NUM_SAMPLES)
    validation_set_size = int(0.1 * NUM_SAMPLES)
    test_set_size = int(0.1 * NUM_SAMPLES)

    random.seed(10)
    random.shuffle(indices)

    training_indices = indices[:train_set_size]
    validation_indices = indices[train_set_size:train_set_size+validation_set_size]
    test_indices = indices[train_set_size+validation_set_size:]

    # reduce training size
    if training_size < 0.8:
        new_train_set_size = int(training_size * NUM_SAMPLES)
        training_indices = training_indices[:new_train_set_size]

    return training_indices, validation_indices, test_indices

def process_csi(csi):
        # concatenate into one 6400x1 vector
        csi = torch.flatten(csi, start_dim=1)
        # split complex values
        csi = torch.stack((csi.real, csi.imag), -1) # vector is now 6400x2
        # concatenate into one 12800x1 vector
        csi = torch.flatten(csi, start_dim=1)

        return csi.float()