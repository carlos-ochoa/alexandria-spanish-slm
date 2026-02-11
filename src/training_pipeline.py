"""Defines the training pipeline"""

import comet_ml
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model, watch

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import AlexandriaDataset
from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import (
    ConfigManager,
    create_collate_fn,
    evaluate,
    generate_text,
    save_model_checkpoint,
)

cm = ConfigManager("config.yaml")
tokenized_tensors_path = cm.config["data"]["tokenized_tensors"]
batch_size = cm.config["batch_size"]
vocab_size = cm.config["model"]["vocab_size"]

hyperparams = cm.config["model"]
log_every = 10
max_tokens = 3
eval_prompt = "Un conejo y una tortuga se encontraban"

tokenizer = AlexandriaTokenizer(load_tokenizer=True)
pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenized_tensors = torch.load(tokenized_tensors_path)
dataset = AlexandriaDataset(tokenized_tensors)
train_data, test_data = train_test_split(dataset, test_size=0.2, train_size=0.8)

train_data = DataLoader(
    train_data,
    batch_size=batch_size,
    #shuffle=True,
    collate_fn=create_collate_fn(pad_token_id=pad_token_id),
)
test_data = DataLoader(
    test_data,
    batch_size=batch_size,
    #shuffle=True,
    collate_fn=create_collate_fn(pad_token_id=pad_token_id),
)

model = AlexandriaModel(config=cm.config)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=100) # Check way of working

progress = tqdm(range(len(train_data)))

experiment = comet_ml.start(
    project="alexandria-slm",
    mode="get_or_create",
    online=True,
    experiment_config=comet_ml.ExperimentConfig(
        auto_log_co2=True, name="small_test_v1", tags=["v1"],
        log_graph=True,
        auto_metric_logging=True,
    ),
)

watch(model)

experiment.log_parameters(hyperparams)

model.to(device)
model.train()

for step, batch in enumerate(train_data):
    input_data = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    #print("INPUT, ", input_data["input_ids"])
    outputs = model(input_data)
    outputs = outputs.view(-1, vocab_size) # equivalent to .view(batch_size * seq_len, vocab_size)
    labels = input_data["labels"].view(-1) # equivalent to .view(batch_size * seq_len,)
    # Fix labels shape to match loss_fn
    loss = loss_fn(outputs, labels)
    experiment.log_metric("train_loss", loss.item(), step=step)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    if step % log_every == 0:
        model.eval()
        eval_metrics = evaluate(model, test_data, pad_token_id, vocab_size)
        generated_text = generate_text(model, eval_prompt, max_tokens, tokenizer)
        generated_text = tokenizer.visualize_tokenization(generated_text)
        experiment.log_metrics(eval_metrics)
        experiment.log_text(generated_text, step=step)
        checkpoint = save_model_checkpoint(model, step, optimizer, loss, eval_metrics, generated_text)
        experiment.log_asset(checkpoint, file_name=checkpoint, step=step)
        model.train()
    progress.update(1)

log_model(experiment, model, model_name="alexandria_v1")

# When loading vocab, must transform to bytes, not only string
# Y la elección de hiperparámetros del optimizer
# Revisar en pizarron cómo el view con los labels termina logrando tensores que encajan
# Agreguemos la intuición matemática del cross-entropy, su relación con perplexity y con la divergencia KL
