import torch
from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import generate_text, ConfigManager

cm = ConfigManager("config.yaml")
tokenizer = AlexandriaTokenizer(load_tokenizer=True)
model = AlexandriaModel(config=cm.config)
model.load_state_dict(torch.load("assets/model-data_comet-torch-model-2500.pth"))

prompt = "Estaban dos personas en "

tokens = generate_text(model, prompt, 50, tokenizer)
print(tokenizer.visualize_tokenization(tokens))