import logging
import sys
from .utils.logger import Formatter
from .training import train
from .test import test
from . import config, scenario, features, test_flag
from .utils.features import compute_features, del_tmp

def main():
    run = test if test_flag else train
    if features:
        logging.info(f'Parameters: {config}')
        compute_features()
        run()
        del_tmp()
    else:
        logging.info(f'Parameters: {config}')
        run()

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