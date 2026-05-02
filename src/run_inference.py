from tqdm import tqdm

import pandas as pd
import torch

from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import ConfigManager, generate_text

cm = ConfigManager("config.yaml")
version = "v2"
tokenizer = AlexandriaTokenizer(load_tokenizer=True, load_path=f"assets/{version}/tokenizer.json")
model = AlexandriaModel(config=cm.config)

# You can load and test different checkpoints by adding their names in this list
checkpoints = []
#checkpoints = [f"assets/{version}/checkpoint-{step}-{step}.pth" for step in range(0, 3000, 1000)]

checkpoints.append(f"assets/{version}/model-data_comet-torch-model-2500-{version}.pth")

results = {"model": [], "prompt": [], "output": [], "max_tokens": [], "temperature": []}

progress = tqdm(range(len(checkpoints)))

for checkpoint in checkpoints:
    if checkpoint == f"assets/{version}/model-data_comet-torch-model-2500-{version}.pth":
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
    else:
        check = torch.load(checkpoint, map_location=torch.device("cpu"))["model_state_dict"]
        print(checkpoint)
        model.load_state_dict(check)

    # Modify this list to test different prompts
    prompts = [
        "¿Cuál fue el resultado de anoche? ",
    ]

    max_tokens = 200
    temperature = 0.5

    for prompt in prompts:
        tokens = generate_text(model, prompt, max_tokens, tokenizer, T=temperature)

        text = tokenizer.visualize_tokenization(tokens)

        print(text)
        print("-------")

        results["model"].append(checkpoint)
        results["prompt"].append(prompt)
        results["output"].append(text)
        results["max_tokens"].append(max_tokens)
        results["temperature"].append(temperature)
    progress.update(1)

results = pd.DataFrame(results)
results.to_csv(f"assets/results_{version}.csv")
