import torch
import torch.nn as nn
import os
import logging
import numpy as np
from importlib import import_module
from torch.utils.data import DataLoader
from torch import optim
from src import dataset, dataset_dir, batch_size, num_workers, network, learning_rate, epochs, device, checkpoint

positions_file = os.path.join(dataset_dir, 'user_positions.npy')
samples_dir = os.path.join(dataset_dir, 'samples')
training_indices = os.path.join(dataset_dir, 'train_indices.npy')
test_indices = os.path.join(dataset_dir, 'test_indices.npy')

def train():

    # Setup dataset and data loader
    dataset_module = import_module('.' + dataset, package='src.datasets')
    Dataset = dataset_module.CSIDataset
    train_dataset = Dataset(positions_file, samples_dir, training_indices)
    test_dataset = Dataset(positions_file, samples_dir, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Setup model and optimizer
    model = import_module('.' + network, package='src.networks').Network()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # from checkpoint
    if checkpoint:
        checkpoint_path = checkpoint
        checkpt = torch.load(checkpoint_path)
        model.load_state_dict(checkpt['model_state_dict'])
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        epoch = checkpt['epoch'] + 1
        # loss = checkpt['loss']
        logging.info(f'Starting training from checkpoint: current epoch = {epoch} ...')
    # from scratch
    else:
        epoch = 0
        checkpoint_path = os.path.join('output', 'models', 'checkpoints', network + '_checkpoint.pt')

    model.to(device)

    # Define the loss as MSE loss
    criterion = nn.MSELoss()
    se_function = nn.MSELoss(reduction='sum')
    ae_function = nn.L1Loss(reduction='sum')

    training_losses = []
    test_losses = []
    for e in range(epoch, epochs):
        model.train()
        training_loss = {'squared_error': 0,'absolute_error': 0}
        test_loss = {'squared_error': 0,'absolute_error': 0}
        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device).float()

            optimizer.zero_grad()
        
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            
            with torch.no_grad():
                squared_error = se_function(output, labels)
                absolute_error = ae_function(output, labels)
                training_loss['squared_error'] += squared_error.item()
                training_loss['absolute_error'] += absolute_error.item()
        else:
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': e,
                'loss': loss.item
            }, checkpoint_path)
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                for samples, labels in test_loader:
                    samples, labels = samples.to(device), labels.to(device)
        
                    output = model(samples)
                    loss = criterion(output, labels.float())

                    test_loss['squared_error'] += squared_error.item()
                    test_loss['absolute_error'] += absolute_error.item()
                
            training_loss = {key: value/len(train_dataset) for key, value in training_loss.items()}
            test_loss = {key: value/len(test_dataset) for key, value in test_loss.items()}
            training_losses.append(training_loss)
            test_losses.append(test_loss)

            logging.info(
                "Epoch: {}/{}... ".format(e+1, epochs) +
                "Training Loss: {:.3f}... ".format(training_loss['absolute_error']) +
                "Test Loss: {:.3f}... ".format(test_loss['absolute_error'])
            )

    # Save the final model
    model_path = os.path.join('output', 'models', network + '.pt')
    torch.save(model.state_dict(), model_path)

    # Save the losses for later analysis
    losses_path = os.path.join('output', 'losses', network + '.pt')
    torch.save({
        'training_losses': training_losses,
        'test_losses': test_losses
    }, losses_path)
