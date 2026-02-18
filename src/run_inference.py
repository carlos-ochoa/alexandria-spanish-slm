import torch
from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import generate_text, ConfigManager

import pandas as pd
from tqdm import tqdm

cm = ConfigManager("config.yaml")
version = "v2"
if version == "v1":
    tokenizer = AlexandriaTokenizer(load_tokenizer=True, load_path="assets/tokenizer_good.json")
else:
    tokenizer = AlexandriaTokenizer(load_tokenizer=True, load_path="assets/tokenizer.json")
model = AlexandriaModel(config=cm.config)

checkpoints = [
    f"assets/{version}/checkpoint-{step}-{step}.pth" for step in range(0, 3000, 1000) 
]

checkpoints.append(f"assets/{version}/model-data_comet-torch-model-2500-{version}.pth")

results = {
    "model" : [],
    "prompt" : [],
    "output" : [],
    "max_tokens" : [],
    "temperature" : []
}

progress = tqdm(range(len(checkpoints)))

for checkpoint in checkpoints:
    if checkpoint == f"assets/{version}/model-data_comet-torch-model-2500-{version}.pth":
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    else:
        check = torch.load(checkpoint, map_location=torch.device('cpu'))["model_state_dict"]
        print(checkpoint)
        model.load_state_dict(check)

    prompts = [
        "Estaban dos personas en ",
        "El conejo estaba a la orilla del río ",
        "La tor",
        "Carlos ",
        "Azul es el cielo ",
        "Moraleja: "
    ]

    max_tokens = 50
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
#print(tokenizer.visualize_tokenization(tokens))

# Hablemos sobre el tema de los sesgos implícitos, y cómo queda de aprendizaje poder analizar mejor los datos
# sintéticos o no, con técnicas como NLP para informar mejor su composición
# Agregar una columna con greedy y otra con Temp