"""Defines the training pipeline"""

import tqdm

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import AlexandriaDataset
from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import ConfigManager, create_collate_fn

cm = ConfigManager("config.yaml")
tokenized_tensors_path = cm.config["data"]["tokenized_tensors"]
batch_size = cm.config["batch_size"]

pad_token_id = AlexandriaTokenizer(load_tokenizer=True).pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenized_tensors = torch.load(tokenized_tensors_path)
dataset = AlexandriaDataset(tokenized_tensors)
train_data, test_data = train_test_split(dataset, test_size=0.2, train_size=0.8)

train_data = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=create_collate_fn(pad_token_id=pad_token_id),
)
test_data = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=create_collate_fn(pad_token_id=pad_token_id),
)

model = AlexandriaModel(config=cm.config)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters())

progress = tqdm(range(train_data.__len__))

model.to(device)

for batch in train_data:
    # Move the model to the GPU (also the data)
    input_data = {k: v.to(device) for k, v in batch}
    # Get the raw logits from the model
    outputs = model(**input_data)
    # Calculate the loss over the outputs
    loss = loss_fn(outputs, input_data["labels"])
    loss.backward()
    optimizer.step()
    progress.update(1)

# No olvidemos los checkpoints