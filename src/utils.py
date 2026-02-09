import yaml

from torch.nn.utils.rnn import pad_sequence

from src.model import AlexandriaModel


class ConfigManager:
    def __init__(self, path: str):
        self.config = self._load_config(path)

    def _load_config(self, path: str) -> dict:
        with open(path, encoding="utf8") as f:
            config = yaml.safe_load(f)
        return config


class EvaluationMetrics:
    def __init__(self):
        pass


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


def evaluate(model: AlexandriaModel):
    model.eval()
    # evaluator = EvaluationMetrics()
    metrics = None
    return metrics
