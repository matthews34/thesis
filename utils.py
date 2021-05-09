import numpy as np
import random
import time
import pandas as pd
import progressbar as pb
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.core.indexes.range import RangeIndex
from datetime import datetime

def running_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def timed_print(*values: object):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string, ' '.join(values))

def load_data(file_path: str) -> object:
    '''Load NPY file'''
    file = open(file_path, 'rb')
    data = np.load(file, allow_pickle=False)
    file.close()
    return data

def load_posistions(root_dir: str) -> pd.DataFrame:
    # Load user possitions
    positions_file = Path(root_dir).joinpath('user_positions.npy')
    positions = load_data(positions_file)
    
    # Set up dataframe
    df = pd.DataFrame(data=positions, columns=['x', 'y', 'z'])
    
    return df

def separate_indices(dataframe: pd.DataFrame, grid_size: int, training_percentage=0.8) -> tuple:
    '''Separe dataset into training set and test set'''
    
    df = dataframe.copy()
    x_array = np.array(df.x)
    y_array = np.array(df.y)

    # Split dataframe into 10 columns (x_bins) and 10 rows (y_bins) (for grid_size = 10)
    x_bins = pd.cut(df.x, np.linspace(min(x_array) - 1, max(x_array) + 1, num=grid_size))
    y_bins = pd.cut(df.y, np.linspace(min(y_array) - 1, max(y_array) + 1, num=grid_size))
    
    # Add column to keep track of indices
    df['i'] = RangeIndex(len(df))
    
    # Group data by bins and set them as index
    df.groupby([x_bins, y_bins])
    df['x_bin'] = x_bins
    df['y_bin'] = y_bins
    df = df.set_index(['x_bin', 'y_bin'])

    # Divide indices into their respective sectors using the bins
    sectors = []
    print('Selecting training set indices')
    for x_bin in pb.progressbar(x_bins.drop_duplicates()):
        for y_bin in y_bins.drop_duplicates():
            df_filtered = df.loc[[(x_bin, y_bin)]]
            sector = df_filtered['i'].tolist()
            sectors.append(sector)

    # Separate training set and test set indices for each sector
    training_set = []
    test_set = []
    random.seed(time.time())
    for sec in sectors:
        training_indices = random.sample(sec, int(len(sec) * training_percentage))
        test_indices = list(set(sec) - set(training_indices))
        training_set += training_indices
        test_set += test_indices

    return training_set, test_set

def separate_dataset(root_dir: Path, training_set_indices, test_set_indices):
    '''Separate dataset into two folders for training and test data'''

    samples_dir = Path.joinpath(root_dir, 'samples')
    training_set_dir = Path.joinpath(samples_dir, 'training_set')
    test_set_dir = Path.joinpath(samples_dir, 'test_set')
    training_set_dir.mkdir(exist_ok=True)
    test_set_dir.mkdir(exist_ok=True)
    print('Separating sets...')
    print('Training set...')
    for index in pb.progressbar(training_set_indices):
        filename = 'channel_measurement_{:06d}.npy'.format(index)
        filepath = Path.joinpath(samples_dir, filename)
        new_filepath = Path.joinpath(training_set_dir, filename)
        Path.rename(filepath, new_filepath)
    print('Test set...')
    for index in pb.progressbar(test_set_indices):
        filename = 'channel_measurement_{:06d}.npy'.format(index)
        filepath = Path.joinpath(samples_dir, filename)
        new_filepath = Path.joinpath(test_set_dir, filename)
        Path.rename(filepath, new_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Separate CSI dataset into training and testing sets')
    parser.add_argument('root_dir', type=str)
    parser.add_argument('grid_size', type=int)
    args = parser.parse_args()
    # dataset_root_dir = Path('/mnt/d/mathe/Documents/thesis/ultra_dense/DIS_lab_LoS')
    dataset_root_dir = Path(args.root_dir)
    grid_size = args.grid_size
    training_set, test_set = separate_indices(dataset_root_dir, grid_size)
    separate_dataset(dataset_root_dir, training_set, test_set)
    print(len(training_set), len(test_set), len(training_set + test_set))