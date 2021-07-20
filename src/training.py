import torch
import torch.nn as nn
import os
import logging
import numpy as np
import torch.nn.functional as F
from importlib import import_module
from torch import optim
from src import dataset_dir, batch_size, num_workers, network, learning_rate, epochs, device, checkpoint
from src.datasets.default import create_dataloader

THRESHOLD = 10

def train():
    training, validation, _ = create_dataloader()

    # Setup model and optimizer
    model_path = os.path.join('output', 'models', network + '.pt')
    model = import_module('.' + network, package='src.networks').Network()
    epoch = 0
    checkpoint_path = os.path.join('output', 'models', 'checkpoints', network + '_checkpoint.pt')

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    for e in range(epoch, epochs):
        model.train()
        training_loss = {'squared_error': 0,'absolute_error': 0, 'distance': 0}
        validation_loss = {'squared_error': 0,'absolute_error': 0, 'distance': 0}
        for samples, labels in training.loader:
            samples, labels = samples.to(device), labels.to(device).float()

            optimizer.zero_grad()
        
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            
            with torch.no_grad():
                squared_error = F.mse_loss(output, labels, reduction='sum')
                absolute_error = F.l1_loss(output, labels, reduction='sum')
                distance = torch.sum(F.pairwise_distance(output, labels))
                training_loss['distance'] += distance.item()
                training_loss['squared_error'] += squared_error.item()
                training_loss['absolute_error'] += absolute_error.item()
        else:
            # Evaluate model
            model.eval()
            with torch.no_grad():
                for samples, labels in validation.loader:
                    samples, labels = samples.to(device), labels.to(device)
        
                    output = model(samples)

                    squared_error = F.mse_loss(output, labels, reduction='sum')
                    absolute_error = F.l1_loss(output, labels, reduction='sum')
                    distance = torch.sum(F.pairwise_distance(output, labels))
                    validation_loss['distance'] += distance.item()
                    validation_loss['squared_error'] += squared_error.item()
                    validation_loss['absolute_error'] += absolute_error.item()
                
            training_loss = {key: value/training.size for key, value in training_loss.items()}
            validation_loss = {key: value/validation.size for key, value in validation_loss.items()}
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            # Save the best model
            if validation_loss['squared_error'] < best_validation_loss:
                best_validation_loss = validation_loss['squared_error']
                torch.save(model.state_dict(), model_path)

            logging.info(
                "Epoch: {}/{}... ".format(e+1, epochs) +
                "Training Loss: {:.3f}... ".format(training_loss['absolute_error']) +
                "Validation Loss: {:.3f}... ".format(validation_loss['absolute_error']) +
                "Validation Distance: {:.3f}".format(validation_loss['distance'])
            )

    # Save the losses for later analysis
    losses_path = os.path.join('output', 'losses', network + '_losses.pt')
    torch.save({
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }, losses_path)
