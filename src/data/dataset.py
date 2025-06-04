import torch
from torch.utils.data import Dataset
import numpy as np
import os

class Semantic3DDataset(Dataset):
    """Semantic3D dataset loader"""
    def __init__(self, data_path, split='train', num_points=4096):
        self.data_path = data_path
        self.num_points = num_points
        self.split = split
        self.class_names = [
            'man-made terrain', 'natural terrain', 'high vegetation',
            'low vegetation', 'buildings', 'hard scape',
            'scanning artifacts', 'cars'
        ]

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass
