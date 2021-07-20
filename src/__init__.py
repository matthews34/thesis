import json
import argparse
import torch
from pathlib import Path

# Parse mutable arguments (passed through command line)
parser = argparse.ArgumentParser(description='Train a neural network.')
parser.add_argument('network', metavar='NETWORK',type=str, help='Name of the network file (without extension)')
parser.add_argument('--scenario', metavar='SCENARIO', type=str, default=None, help='Basename to use when saving output (default is network name)')
parser.add_argument('--lr', metavar='0.001', type=float, default=0.001, help='Learning rate of the training algorithm')
parser.add_argument('--epochs', metavar='200', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch', metavar='64', type=int, default=64, help='Size of the training and validation batches')
parser.add_argument('--num_workers', metavar='8', type=int, default=8, help='Number of workers for mutithread loading')
parser.add_argument('--training_size', metavar='0.8', type=float, default=0.8, help='Portion of the dataset corresponding to the training set')
parser.add_argument('--config_file', metavar='PATH_TO_CONFIG', type=str, default='config.json', help='Path to config file')

args = parser.parse_args()
network = args.network
scenario = args.scenario if args.scenario else network
learning_rate = args.lr
epochs = args.epochs
batch_size = args.batch
num_workers = args.num_workers
training_size = args.training_size
config_file = args.config_file

Path('output/logs').mkdir(parents=True, exist_ok=True)
Path('output/losses').mkdir(parents=True, exist_ok=True)
Path('output/models').mkdir(parents=True, exist_ok=True)

with open(config_file, 'r') as file:
    config = json.load(file)

dataset_dir = config['dataset_dir']

if 'device' in config.keys():
    device = torch.device(config['device'])
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'network': network,
    'scenario name': scenario,
    'learning rate': learning_rate, 
    'epochs': epochs, 
    'batch size': batch_size, 
    'num workers': num_workers, 
    'config file': config_file, 
    'training size': training_size,
    'device': device.type,
}