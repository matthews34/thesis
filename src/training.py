import torch
import torch.nn as nn
import os
import logging
import numpy as np
import torch.nn.functional as F
from importlib import import_module
from torch import optim
from src import dataset_dir, batch_size, num_workers, network, learning_rate, epochs, device, scenario, dataset_name


from src.datasets.default import create_dataloader

def train():
    dataset = import_module('src.datasets.{}'.format(dataset_name))
    training, validation, _ = dataset.create_dataloader()

    # Setup model and optimizer
    scenario_name = os.path.join('output', 'models', scenario + '.pt')
    model = import_module('.' + network, package='src.networks').Network()
    epoch = 0
    checkpoint_path = os.path.join('output', 'models', 'checkpoints', network + '_checkpoint.pt')

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    times = []
    best_validation_loss = float('inf')
    for e in range(epoch, epochs):
        model.train()
        training_loss = {'squared_error': torch.empty((0, 2), device=device),'absolute_error': torch.empty((0, 2), device=device), 'distance': torch.empty((0, 1), device=device)}
        validation_loss = {'squared_error': torch.empty((0, 2), device=device),'absolute_error': torch.empty((0, 2), device=device), 'distance': torch.empty((0, 1), device=device)}

        # Timing
        if device == torch.device('cuda'):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

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
                training_loss['squared_error'] = torch.cat((training_loss['squared_error'], F.mse_loss(output, labels, reduction='none')), dim=0)
                training_loss['absolute_error'] = torch.cat((training_loss['absolute_error'], F.l1_loss(output, labels, reduction='none')), dim=0)
                training_loss['distance'] = torch.cat((training_loss['distance'], F.pairwise_distance(output, labels, keepdim=True)), dim=0)

        else:
            # Timing
            time = None
            if device == torch.device('cuda'):
                end.record()
                torch.cuda.synchronize()
                time = start.elapsed_time(end)
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                for samples, labels in validation.loader:
                    samples, labels = samples.to(device), labels.to(device)
        
                    output = model(samples)

                    validation_loss['squared_error'] = torch.cat((validation_loss['squared_error'], F.mse_loss(output, labels, reduction='none')), dim=0)
                    validation_loss['absolute_error'] = torch.cat((validation_loss['absolute_error'], F.l1_loss(output, labels, reduction='none')), dim=0)
                    validation_loss['distance'] = torch.cat((validation_loss['distance'], F.pairwise_distance(output, labels, keepdim=True)), dim=0)

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            # Save the best model
            if torch.mean(validation_loss['squared_error']) < best_validation_loss:
                best_validation_loss = torch.mean(validation_loss['squared_error']).item()
                torch.save(model.state_dict(), scenario_name)

            if time:
                times.append(time)
                logging.debug('Time: {} ms'.format(time))
            logging.info(
                "Epoch: {}/{}... ".format(e+1, epochs) +
                "Training Loss: {:.3f}... ".format(torch.mean(training_loss['absolute_error']).item()) +
                "Validation Loss: {:.3f}... ".format(torch.mean(validation_loss['absolute_error']).item()) +
                "Validation Distance: {:.3f}".format(torch.mean(validation_loss['distance']).item())
            )

    # Save losses for later analysis
    losses_path = os.path.join('output', 'losses', scenario + '_losses.pt')
    save_dict = {
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'times': times,
    }
    torch.save(save_dict, losses_path)
