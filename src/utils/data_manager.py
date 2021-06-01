import numpy as np
import os
from importlib import import_module
from torch.utils.data import DataLoader
from src import dataset_dir, dataset, batch_size, num_workers

def load_data(file_path: str) -> object:
    '''Load NPY file'''
    file = open(file_path, 'rb')
    data = np.load(file, allow_pickle=False)
    file.close()
    return data

def create_dataloader():
    positions_file = os.path.join(dataset_dir, 'user_positions.npy')
    samples_dir = os.path.join(dataset_dir, 'samples')
    training_indices = os.path.join(dataset_dir, 'train_indices.npy')
    test_indices = os.path.join(dataset_dir, 'test_indices.npy')

    # Setup dataset and data loader
    dataset_module = import_module('.' + dataset, package='src.datasets')
    Dataset = dataset_module.CSIDataset
    training_set = Dataset(positions_file, samples_dir, training_indices)
    test_set = Dataset(positions_file, samples_dir, test_indices)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return training_loader, test_loader, len(training_set), len(test_set)