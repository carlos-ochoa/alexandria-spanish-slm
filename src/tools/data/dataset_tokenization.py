"""This script contains the initial flow for preparing the dataset processed for training
"""

from collections import Counter
from datasets import load_dataset
import unicodedata
from tokenizer import AlexandriaTokenizer

def clean_data(row : str) -> str:
  row["text"] = unicodedata.normalize('NFKD', row["text"])
  return row

ds = load_dataset("wikimedia/wikipedia", "20231101.es")
#ds = ds.map(clean_data)

dataset = ds['train'].shuffle(seed=42).select(range(1000))
dataset = list(dataset["text"])
tokenizer_ds = ds['train'].shuffle(seed=42).select(range(35))
tokenizer_ds = tokenizer_ds["text"]

tokenizer = AlexandriaTokenizer()
tokenizer.build_vocab(tokenizer_ds)

tokenized_dataset = tokenizer.tokenize(dataset)

compression_ratio_char_level = 0
for original_text, tokenized_text in zip(tokenizer_ds, tokenized_dataset):
    compression_ratio_char_level += len(original_text) / len(tokenized_text)
compression_ratio_char_level /= len(tokenized_dataset)

compression_ratio_byte_level = 0
for original_text, tokenized_text in zip(tokenizer_ds, tokenized_dataset):
    compression_ratio_byte_level += len(original_text.encode('utf-8')) / len(tokenized_text)
compression_ratio_byte_level /= len(tokenized_dataset)

print(f"compression_ratio_char_level: {compression_ratio_char_level}")
print(f"compression_ratio_byte_level: {compression_ratio_byte_level}")

print("Most frequent merges in AlexandriaTokenizer")
print(tokenizer.most_frequent_merges.most_common(20))
for merge, _ in tokenizer.most_frequent_merges.most_common(20):
   token_merge = tokenizer.merges[merge[0]]
   print(token_merge , tokenizer.decode([token_merge]))

print("Most frequent tokens discovered in unseen data")
c = Counter()
for t in tokenized_dataset:
    c.update(t)
for token in c.most_common(20):
   print(tokenizer.decode(token))
