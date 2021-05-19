import logging
import sys
from .utils.logger import Formatter
from .training import train
from . import config, network

def main():
    logging.debug('Calling train()...')
    logging.info(f'Parameters: {config}')
    train()

def test():
    pass

if __name__ == '__main__':
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(Formatter())
    consoleHandler.setLevel(logging.DEBUG)
    
    fileHandler = logging.FileHandler(f'output/logs/{network}.log')
    fileHandler.setFormatter(Formatter())
    fileHandler.setLevel(logging.INFO)

    logging.root.addHandler(consoleHandler)
    logging.root.addHandler(fileHandler)
    logging.root.setLevel(logging.DEBUG)

    main()