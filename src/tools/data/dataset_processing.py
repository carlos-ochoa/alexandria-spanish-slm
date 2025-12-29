"""This script contains the initial flow for preparing the dataset processed for training
"""

from datasets import load_dataset
import unicodedata
from tokenizer import AlexandriaTokenizer

def clean_data(row : str) -> str:
  row["text"] = unicodedata.normalize('NFKD', row["text"])
  return row

ds = load_dataset("wikimedia/wikipedia", "20231101.es")
ds = ds.map(clean_data)
ds = ds["train"]["text"]

tokenizer_ds = ds['train'].shuffle(seed=42).select(range(200000))
tokenizer_ds = tokenizer_ds["text"]

tokenizer = AlexandriaTokenizer()
tokenizer.build_vocab(tokenizer_ds)

tokenized_dataset = tokenizer.tokenize(ds)

compression_ratio_char_level = 0
for original_text, tokenized_text in zip(tokenizer_ds, tokenized_dataset):
    compression_ratio_char_level += len(original_text) / len(tokenized_text)
compression_ratio_char_level /= len(tokenized_dataset)

compression_ratio_byte_level = 0
for original_text, tokenized_text in zip(tokenizer_ds, tokenized_dataset):
    compression_ratio_byte_level += len(original_text.encode('utf-8')) / len(tokenized_text)
compression_ratio_byte_level /= len(tokenized_dataset)

print(compression_ratio_char_level)
print(compression_ratio_byte_level)

# Calculate the compression reason for the tokenizer
# Visualize the top 20 most seen tokens for my tokenizer




