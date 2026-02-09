"""Defines the dataset for Alexandria"""

from torch.utils.data import Dataset


class AlexandriaDataset(Dataset):
    def __init__(self, tokenized_dataset: list):
        self.tokenized_dataset = tokenized_dataset["tokenized_docs"]

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        return self.tokenized_dataset[index][:-1], self.tokenized_dataset[index][1:]
