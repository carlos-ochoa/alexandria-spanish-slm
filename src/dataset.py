"""Defines the dataset for Alexandria"""

from torch.utils.data import Dataset


class AlexandriaDataset(Dataset):
    def __init__(self, tokenized_dataset: list):
        self.tokenized_dataset = tokenized_dataset["tokenized_docs"]

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # I have to return the input text and the target label
        # Remember I can build many examples from the training data
        # Is this place the best to perform that operation? Considering I'm returning single examples here
        return self.tokenized_dataset[index][:-1], self.tokenized_dataset[index][1:]
