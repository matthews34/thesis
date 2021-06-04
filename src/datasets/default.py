import os
import numpy as np
from torch.utils.data import Dataset
from src.utils.data_manager import load_data

# Configure dataset
class CSIDataset(Dataset):
    """CSI dataset."""
    
    def __init__(self, positions_file, samples_dir, indices):
        """
        Args:
            positions_file (string): Path to the file containing the user positions.
            samples_dir (string): Directory containing the samples.
            indexes_file (string): Path to the file holding the indexes to be considered for the set
        """
        self.user_positions = load_data(positions_file)
        self.samples_dir = samples_dir
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        
        sample_filepath = os.path.join(self.samples_dir, 'channel_measurement_{:06d}.npy'.format(index))
        sample = load_data(sample_filepath)
                    
        # Remove z coordinate from the positions
        label = np.delete(self.user_positions[index], -1)
        
        return sample, label
