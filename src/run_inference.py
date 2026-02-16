import torch
from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import generate_text, ConfigManager

import pandas as pd
from tqdm import tqdm

cm = ConfigManager("config.yaml")
tokenizer = AlexandriaTokenizer(load_tokenizer=True)
model = AlexandriaModel(config=cm.config)

#checkpoints = [
#    f"assets/checkpoint-{step}-{step}.pth" for step in range(0, 2500, 100) 
#]
checkpoints = []

checkpoints.append("assets/model-data_comet-torch-model-2500-v2.pth")

results = {
    "model" : [],
    "prompt" : [],
    "output" : [],
    "max_tokens" : [],
    "temperature" : []
}

progress = tqdm(range(len(checkpoints)))

for checkpoint in checkpoints:
    if checkpoint == "assets/model-data_comet-torch-model-2500-v2.pth":
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    else:
        check = torch.load(checkpoint)["model_state_dict"]
        model.load_state_dict(check)

    prompts = [
        "Estaban dos personas en ",
        "El conejo estaba a la orilla del río ",
        "La tor",
        "Carlos ",
        "Azul es el cielo ",
        "Sin recursos moriremos "
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
results.to_csv("assets/results_v2.csv")
#print(tokenizer.visualize_tokenization(tokens))

# Hablemos sobre el tema de los sesgos implícitos, y cómo queda de aprendizaje poder analizar mejor los datos
# sintéticos o no, con técnicas como NLP para informar mejor su composición
# Agregar una columna con greedy y otra con Temp