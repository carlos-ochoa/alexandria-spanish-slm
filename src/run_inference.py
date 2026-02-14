import torch
from src.model import AlexandriaModel
from src.tools.data.tokenizer import AlexandriaTokenizer
from src.utils import generate_text, ConfigManager

import pandas as pd

cm = ConfigManager("config.yaml")
tokenizer = AlexandriaTokenizer(load_tokenizer=True)
model = AlexandriaModel(config=cm.config)

checkpoints = [
    f"assets/checkpoint-{step}-{step}.pth" for step in range(0, 2500, 100) 
]
checkpoints.append("assets/model-data_comet-torch-model-2500.pth")

results = {
    "model" : [],
    "prompt" : [],
    "output" : [],
    "max_tokens" : [],
    "temperature" : []
}

for checkpoint in checkpoints:
    model.load_state_dict(torch.load(checkpoint))

    prompts = [
        "Estaban dos personas en ",
        "El conejo estaba a la orilla del río ",
        "La tor",
        "Carlos ",
        "Azul es el cielo "
    ]

    max_tokens = 50
    temperature = 0.5

    for prompt in prompts:
        tokens = generate_text(model, prompt, max_tokens, tokenizer, T=temperature)

    results["model"].append(checkpoint)
    results["prompt"].append(prompt)
    results["output"].append(tokens)
    results["max_tokens"].append(max_tokens)
    results["temperature"].append(temperature)

results = pd.DataFrame(results)
results.to_csv("assets/results_v2.csv")
#print(tokenizer.visualize_tokenization(tokens))

# Hablemos sobre el tema de los sesgos implícitos, y cómo queda de aprendizaje poder analizar mejor los datos
# sintéticos o no, con técnicas como NLP para informar mejor su composición