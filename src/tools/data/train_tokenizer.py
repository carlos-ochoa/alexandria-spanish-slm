"""This script contains the initial flow for preparing the dataset processed for training"""

from datasets import load_dataset

from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import ConfigManager


cm = ConfigManager("config.yaml")
tokenized_dataset_path = cm.config["data"]["tokenized_tensors"]

ds = load_dataset("hetline/tiny-coop-es")

tokenizer_ds = ds["train"].shuffle(seed=42).select(range(1500))
tokenizer_ds = tokenizer_ds["text"]

tokenizer = AlexandriaTokenizer()
tokenizer.build_vocab(tokenizer_ds)
tokenizer.save_tokenizer()