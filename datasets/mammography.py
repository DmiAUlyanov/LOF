from torch.utils.data import Dataset
from pathlib import Path
import scipy.io

import config


class MammographyDataset(Dataset):
    r"""
    http://odds.cs.stonybrook.edu/mammography-dataset/
    """

    def __init__(self, data_path: Path = config.DATA_PATH):
        mat = scipy.io.loadmat(str(data_path / 'mammography.mat'))
        self.vectors, self.labels = mat['X'], mat['y']

    def __len(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
