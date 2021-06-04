import numpy as np
import os
import random
from types import SimpleNamespace
from importlib import import_module
from torch.utils.data import DataLoader
from src import dataset_dir, dataset, batch_size, num_workers

NUM_SAMPLES = 252004

def load_data(file_path: str) -> object:
    '''Load NPY file'''
    file = open(file_path, 'rb')
    data = np.load(file, allow_pickle=False)
    file.close()
    return data

def create_dataloader():
    positions_file = os.path.join(dataset_dir, 'user_positions.npy')
    samples_dir = os.path.join(dataset_dir, 'samples')
    training_indices, validation_indices, test_indices = generate_indices()
    # training_indices = os.path.join(dataset_dir, 'train_indices.npy')
    # test_indices = os.path.join(dataset_dir, 'test_indices.npy')

    # Setup dataset and data loader
    dataset_module = import_module('.' + dataset, package='src.datasets')
    Dataset = dataset_module.CSIDataset

    training_set = Dataset(positions_file, samples_dir, training_indices)
    validation_set = Dataset(positions_file, samples_dir, validation_indices)
    test_set = Dataset(positions_file, samples_dir, test_indices)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    training = SimpleNamespace(**{'loader': training_loader, 'size': len(training_set)})
    validation = SimpleNamespace(**{'loader': validation_loader, 'size': len(validation_set)})
    test = SimpleNamespace(**{'loader': test_loader, 'size': len(test_set)})

    return training, validation, test

def generate_indices():
    indices = [x for x in range(NUM_SAMPLES)]

    training_size = int(0.8 * NUM_SAMPLES)
    validation_size = int(0.1 * NUM_SAMPLES)
    test_size = int(0.1 * NUM_SAMPLES)

    random.seed(10)
    random.shuffle(indices)

    training_indices = indices[:training_size]
    validation_indices = indices[training_size:training_size+validation_size]
    test_indices = list(set(indices) - set(training_indices) - set(validation_indices))

    return training_indices, validation_indices, test_indices

