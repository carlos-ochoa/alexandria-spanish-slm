"""Defines the training pipeline"""

import comet_ml
import tqdm
from comet_ml.integration.pytorch import log_model, watch

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

hyperparams = cm.config["model"]
log_every = 100

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
watch(model)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters())

progress = tqdm(range(train_data.__len__))

experiment = comet_ml.start(
    project="alexandria-slm",
    mode="get_or_create",
    online=True,
    log_graph=True,
    auto_metric_logging=True,
    experiment_config=comet_ml.ExperimentConfig(
        auto_log_co2=True, name="small_test_v1", tags=["v1"]
    ),
)

experiment.log_parameters(hyperparams)

model.to(device)
model.train()

for step, batch in enumerate(train_data):
    input_data = {k: v.to(device) for k, v in batch}
    optimizer.zero_grad()
    outputs = model(**input_data)
    loss = loss_fn(outputs, input_data["labels"])
    experiment.log_metric("train_loss", loss.item(), step=step)
    # Evaluate the current behavior of the model
    if step % log_every == 0:
        pass
    loss.backward()
    optimizer.step()
    progress.update(1)

log_model(experiment, model, model_name="alexandria_v1")

# No olvidemos los checkpoints
# Y la elección de hiperparámetros del optimizer
# Agreguemos un scheduler
# Agregar en evaluation que se generen ejemplos de generación cada ciertos steps para ver cómo evoluciona
# Considerar un early stopping
# Arreglar el acceso a configuraciones desde el model.py
