# Alexandria's model design choices

This document discusses the design choices for the model's architecture.

## Relevant sources
As part of the design process some relevant papers have been analyzed. Even though not all the lessons have been applied they are still an incredible reference for future decisions.

- [Attention is all you need](https://arxiv.org/pdf/1706.03762)
- [Training compute-optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
- [Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361)
- [TinyStories: How small can language models be and still speak coherent English?](https://arxiv.org/pdf/2305.07759)

## Model Architecture

Alexandria is implemented as a decoder-only transformer model. It is primarily based on the original architecture presented in [Attention is all you need](https://arxiv.org/pdf/1706.03762). However, also some other design choices were made specifically to the goals for Alexandria.

### Architecture Diagram

The diagram has been built using Excalidraw.

![alt text](img/architecture.png)

- Total parameters: ≈ 9M
- Transformer layers: 6
- Attention heads per transformer layer: 8
- Context window size: 256

The decision to make the model of a size of 9M parameters relies in the next ideas:
- Empirically, we can see learned language patterns with architectures with more than 5M-10M parameters. 
- As discussed in [Training compute-optimal Large Language Models](https://arxiv.org/pdf/2203.15556) it is expected to train a compute-optimal model by training on 20B per 1B parameters. However, Alexandria's size is considerably smaller than that. Hence, I decided to have a 1:1 relation in number of parameters and tokens.
- Alexandria's Tokenizer has an observed compression ratio of around 3.5. Hence, to build a training dataset with 9M tokens I'd need to process around 32M raw tokens. Which with the current implementation of the tokenizer would take around 2 hours.

### Positional Embeddings

For this version, it's been decided to use learnable positional embeddings, fixed with the context window of 256 tokens. This approach has been chosen given the simplicity of the methodology, also considering the presence of enough tokens in the training dataset to perform well-learned positional representations. 

Learning the positional representations is preferred given the specific use case Alexandria is trained on, also inspired by the practical results observed in architectures such as GPT-2.

### Compute and memory estimation

To estimate the amount of compute and memory needed for a performant training of this architecture let's detail the calculations starting with the estimated FLOPs required to train the model.

For training, we'll use FP32 precision.

$FLOPs ≈ 6 × N × D$

Where: 
- N = Number of parameters in the model
- D = Number of tokens in training dataset

$FLOPs ≈ 6 × 9M × 9M = 486B FLOPs = 486 × 10^12 FLOPs = 486 TFLOPs$

Using a RTX 4090 aprox 80 TFLOPs

486 TFLOPs / 80 TFLOPs = 6.075 hours -> Let's asume an upper bound of 2x times the time due to underutilization of the GPU -> Between 6 and 12 hours for training.

In RunPod RTX 4090 costs 0.59 USD per hour. Hence total training might cost aorund 7.08 USD. This costs considers a single epoch.

Compare with RTX 5090

$Memory ≈ 4 × params = 4 x 9M = 36MB$

TODO: 
1. Documentar las decisiones de la arquitectura de la red
2. Armar el script para tener listos los ejemplos de entrenamiento
3. Armar el modulo de IaC para levantar el cluster de runpod
4. Configurar el wandb
4. Configurar el setup para medir impacto ambiental del entrenamiento
5. Documentar todo el training pipeline y las decisiones de diseño
6. Conseguir más tokens
7. Ejecutar el entrenamiento
8. Armar el script de inferencia
9. Configurar para compatibilidad con HuggingFace
10. Puedo tomar solo articulos pequeños completos de menos de 256 tokens, y analizar los topicos. Alexandria seria bueno en eso y su vocab se veria afectado por este tipo de tópicos.

# No lo haré ahorita, pero sería interesante un analisis de cómo las scaling laws permiten tomar decisiones informadas

## Manejo de Padding en el Entrenamiento

Alexandria utiliza `ignore_index` en la función de pérdida para 
evitar que las posiciones de padding contribuyan al gradiente:
```python
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
```

Esto garantiza que el modelo solo aprende a predecir tokens reales 
del español, no a replicar secuencias de padding.

Hablemos del tema de broadcasting también

## Lecciones Aprendidas: Máscaras de Atención

Durante la implementación del sistema de máscaras, invertí tiempo 
significativo entendiendo los fundamentos de broadcasting en PyTorch 
y cómo se combinan múltiples máscaras.

### Broadcasting: El mecanismo clave
[Tus visualizaciones y ejemplos]

### Por qué DOS máscaras
1. Causal mask: Previene atención a futuro (arquitectura del modelo)
2. Padding mask: Previene atención a tokens artificiales (datos de entrenamiento)

### Tradeoff: Corrección vs Eficiencia
La implementación actual computa attention para queries de padding 
(~30% desperdicio computacional) pero garantiza corrección mediante 
`ignore_index` en la loss. Versiones futuras pueden optimizar esto 
con Flash Attention o packed sequences.