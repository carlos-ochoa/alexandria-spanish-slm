import yaml
import time
from tqdm import tqdm

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


def create_collate_fn(pad_token_id=258, max_seq_len=256):
    def custom_padding_collate(batch):
        tensors = [torch.tensor(input_id).long() for input_id, _ in batch]
        labels = [torch.tensor(label).long() for _, label in batch]

        longest_input = max(t.size(0) for t in tensors)

        if longest_input > max_seq_len:
            tensors = [t[:max_seq_len] for t in tensors]
            labels = [t[:max_seq_len] for t in labels]

        padded_batch = pad_sequence(tensors, batch_first=True, padding_value=pad_token_id)

        attention_mask = (padded_batch != pad_token_id).long()

        padded_labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

        return {
            "input_ids": padded_batch,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    return custom_padding_collate


def evaluate(model: AlexandriaModel, test_data: DataLoader, pad_token_id: int, vocab_size: int):
    metrics = {}
    loss = 0
    progress = tqdm(range(len(test_data)))
    with torch.no_grad():
        for batch in test_data:
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            outputs = model(batch)
            outputs = outputs.view(-1, vocab_size)
            labels = batch["labels"].view(-1)
            loss += loss_fn(outputs, labels)
            progress.update(1)
    metrics["test_loss"] = loss / len(test_data)
    metrics["perplexity"] = torch.exp(loss / len(test_data))
    return metrics


def generate_text(
    model: AlexandriaModel, eval_prompt: str, max_tokens: int, tokenizer: AlexandriaTokenizer
):
    tokenized_prompt = tokenizer.tokenize(eval_prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            tokenized_prompt = torch.tensor(tokenized_prompt)
            tokenized_prompt = tokenized_prompt.unsqueeze(0)
            input = {"input_ids": tokenized_prompt, "attention_mask": None}
            start_time = time.time()
            print(start_time)
            output = model(input)
            print("Time per token (s): ", time.time() - start_time)
            next_token_logits = output[
                :, -1, :
            ]  # because we want only the logits for the last token
            next_token = torch.argmax(
                next_token_logits, dim=-1
            )  # current implementation takes greedy decoding
            tokenized_prompt = tokenized_prompt.tolist()[0] + next_token.tolist()
    return tokenized_prompt


def save_model_checkpoint(
    model: AlexandriaModel,
    step: int,
    optimizer,
    loss: float,
    eval_metrics: dict,
    generated_text: str,
):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "eval_metrics": eval_metrics,
            "generated_text": generated_text,
        },
        f"checkpoint-{step}.pth",
    )
    return f"checkpoint-{step}.pth"
