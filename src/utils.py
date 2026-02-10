import yaml

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer


class ConfigManager:
    def __init__(self, path: str):
        self.config = self._load_config(path)

    def _load_config(self, path: str) -> dict:
        with open(path, encoding="utf8") as f:
            config = yaml.safe_load(f)
        return config

def create_collate_fn(pad_token_id=258):
    def custom_padding_collate(batch):
        tensors = [input_id for input_id, _ in batch]
        labels = [label for _, label in batch]

        padded_batch = pad_sequence(tensors, batch_first=True, padding_value=pad_token_id)

        attention_mask = (padded_batch != pad_token_id).long()

        padded_labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

        return {
            "input_ids": padded_batch,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    return custom_padding_collate


def evaluate(model: AlexandriaModel, test_data : DataLoader, pad_token_id : int):
    metrics = {}
    # Calculate the loss on validation data
    with torch.no_grad():
        for batch in test_data:
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            outputs = model(**batch)
            loss = loss_fn(outputs, batch['labels'])
            perplexity = torch.exp(loss)
    metrics['test_loss'] = loss
    metrics['perplexity'] = perplexity
    return metrics

def generate_text(model : AlexandriaModel, eval_prompt : str, max_tokens : int, tokenizer : AlexandriaTokenizer):
    with torch.no_grad():
        for _ in range(max_tokens):
            tokenized_prompt = tokenizer.tokenize(eval_prompt)
            input = {
                'input_ids' : tokenized_prompt,
                'attention_mask' : None
            }
            output = model(**input)
            output = torch.argmax(output)
            eval_prompt = tokenizer.decode(tokenized_prompt + [output])
    return tokenizer.decode(eval_prompt)