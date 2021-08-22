import logging
import sys
from .utils.logger import Formatter
from .training import train
from . import config, scenario, features

from .utils.features import gen_PDP, tof, rss, power_first_path, delay_spread
from .datasets.default import create_dataloader, CSIDataset, generate_indices
import os
from . import dataset_dir
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
torch.set_printoptions(precision=6)

def main():
    logging.info(f'Parameters: {config}')
    train()

if __name__ == '__main__':
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(Formatter())
    consoleHandler.setLevel(logging.DEBUG)
    
    fileHandler = logging.FileHandler(f'output/logs/{scenario}.log')
    fileHandler.setFormatter(Formatter())
    fileHandler.setLevel(logging.INFO)

    logging.root.addHandler(consoleHandler)
    logging.root.addHandler(fileHandler)
    logging.root.setLevel(logging.DEBUG)

    main()