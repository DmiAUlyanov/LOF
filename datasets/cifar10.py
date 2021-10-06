from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from pathlib import Path
import numpy as np

import config


class Cifar10(Dataset):
    r"""
    https://www.cs.toronto.edu/~kriz/cifar.html
    This class is a wrapper over the default pytorch class for ease of use for the anomaly detection task.
    Parameter 'anomaly_class' is responsible for which class will be considered anomalous, while the rest are normal.
    Available classes:
                     'airplane'
                     'automobile'
                     'bird'
                     'cat'
                     'deer'
                     'dog'
                     'frog'
                     'horse'
                     'ship'
                     'truck'
    """
    DEFAULT_DATA_PATH = config.DATA_PATH / 'cifar10'

    def __init__(self, anomaly_class: str, data_path: Path = DEFAULT_DATA_PATH):
        _dataset = CIFAR10(root=str(data_path), download=True, transform=ToTensor())
        anomaly_class_idx = _dataset.class_to_idx[anomaly_class]

        self.vectors = _dataset.data.reshape(_dataset.data.shape[0], -1)
        self.labels = (np.array(_dataset.targets) == anomaly_class_idx).astype(int)

    def __len(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
