"""Defines the dataset for Alexandria
"""

import torch
from torch.utils.data import Dataset

class WikipediaDataset(Dataset):

    def __init__(self, tokenized_dataset_path : str):
        self.tokenized_dataset = torch.load(tokenized_dataset_path)

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        pass