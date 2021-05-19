import numpy as np

def load_data(file_path: str) -> object:
    '''Load NPY file'''
    file = open(file_path, 'rb')
    data = np.load(file, allow_pickle=False)
    file.close()
    return data