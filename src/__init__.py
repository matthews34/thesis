import json
import torch
from pathlib import Path

CONFIG_FILE = ('config.json')

Path('output/logs').mkdir(parents=True, exist_ok=True)
Path('output/losses').mkdir(parents=True, exist_ok=True)
Path('output/models').mkdir(parents=True, exist_ok=True)
Path('output/models/checkpoints').mkdir(parents=True, exist_ok=True)

with open(CONFIG_FILE, 'r') as file:
    config = json.load(file)

network = config['network']
dataset_dir = config['dataset_dir']
epochs = config['epochs']
learning_rate = config['learning_rate']
batch_size = config['batch_size']
num_workers = config['num_workers']
dataset = config['dataset']

if 'device' in config.keys():
    device = torch.device(config['device'])
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if 'checkpoint' in config.keys():
    checkpoint = config['checkpoint']
else:
    checkpoint = None