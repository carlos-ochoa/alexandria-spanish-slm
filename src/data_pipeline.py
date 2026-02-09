"""Prepares the data to fit into the expectations on context window"""

import torch


class DataProcessor:
    def __init__(self, tokenized_dataset_path: str, context_window: int):
        self.tokenized_dataset = torch.load(tokenized_dataset_path)
        self.context_window = context_window

    def run(self):
        # Debemos preparar todos los ejemplos para que sean de máximo el tamaño de la context window
        # Tal vez la mejor manera es en los párrafos.

        pass
