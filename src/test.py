import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
from importlib import import_module
from src import dataset_dir, batch_size, num_workers, network, learning_rate, epochs, device, scenario, dataset_name

def test():
    dataset = import_module('src.datasets.{}'.format(dataset_name))
    _, _, test_set = dataset.create_dataloader()

    # Setup model
    scenario_name = os.path.join('output', 'models', scenario + '.pt')
    model = import_module('.' + network, package='src.networks').Network()
    epoch = 0

    model.to(device)
    model.load_state_dict(torch.load(scenario_name, map_location=device))

    model.eval()
    
    loss = {'squared_error': torch.empty((0, 2), device=device),'absolute_error': torch.empty((0, 2), device=device), 'distance': torch.empty((0, 1), device=device)}

    # Timing
    if device == torch.device('cuda'):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with torch.no_grad():
        for samples, labels in test_set.loader:
            samples, labels = samples.to(device), labels.to(device).float()

            output = model(samples)
            loss['squared_error'] = torch.cat((loss['squared_error'], F.mse_loss(output, labels, reduction='none')), dim=0)
            loss['absolute_error'] = torch.cat((loss['absolute_error'], F.l1_loss(output, labels, reduction='none')), dim=0)
            loss['distance'] = torch.cat((loss['distance'], F.pairwise_distance(output, labels, keepdim=True)), dim=0)
    
    # Timing
    time = None
    if device == torch.device('cuda'):
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        logging.debug('Time: {} s'.format(time/1000))

    logging.info('MAE: {:.3f}... '.format(loss['absolute_error'].mean()) + 'Mean distance: {:.3f}'.format(loss['distance'].mean()))
    loss_path = os.path.join('output', 'tests', scenario + '_test.pt')
    torch.save({'loss': loss, 'time': time}, loss_path)
