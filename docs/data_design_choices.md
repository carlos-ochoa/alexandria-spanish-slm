# Data design choices

This space discuss the design choices for the dataset used during training, preprocessing and tokenization.

---

## Data sources

Alexandria is a small language model specifically trained in Spanish which goal is to generate plausible text with no further capabilities in mind. 

For this project, the only considered data source takes Wikipedia articles in Spanish. The main source comes from the official [Wikimedia dataset in Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.es?row=0&views%5B%5D=_20231101ab). 

As additional information on how the training data looks like. We've saved considerable time by using this official Wikimedia's dataset since it is built over the dumps from Wikipedia and all the wiki's parsing has been already performed, giving us the raw text to work with.

For the training we are considering 1.8M articles in Spanish, with a knowledge cutoff from November 2023.

Alexandria is a small language model that aims to have a communication style similar to the ones found in academy or enciclopedies, hence, using Wikipedia as our main data source makes sense for the desired style.

### Biases and ethical considerations

Alexandria is built mainly as a learning project, hence, I'm not performing extensive analysis or bias considerations for the data curation. 

**Assumptions:**
- Wikipedia articles provide sufficient diversity to represent 
  Spanish language for basic text generation tasks
- Simple random sampling is adequate for tokenizer training
  
**Known Limitations:**
- Temporal bias: Knowledge cutoff at November 2023
- Topical bias: Wikipedia's inherent skew toward certain domains
- Style bias: Formal/encyclopedic register, limited colloquial language
- Demographic bias: Wikipedia editor demographics may influence content

These limitations are acknowledged but not addressed in this iteration. Dealing with a well-crafted dataset is out of the scope for this project.

## Tokenization

Alexandria uses a custom BPE Tokenizer as default tokenizer. As part of this project, I decided to implement the tokenizer myself. 

- **Data size**: A subsample of 200,000 articles will be used to build the vocabulary. Random sampling will be used to pick the articles.
- **Vocabulary size**: 24,000 tokens.

### Special Tokens

Alexandria uses a minimal set of special tokens:

- `<EOS>`: Marks the end of each Wikipedia article
- `<UNK>`: Reserved for unknown sequences (rarely used in byte-level BPE)

**Design rationale**: 
A minimal token set keeps the model simple and allocates more vocabulary 
space to Spanish language tokens. The model learns text structure 
(paragraphs, sections) implicitly from the natural formatting of 
Wikipedia articles.

**Future considerations**: 
Additional control tokens (e.g., `<BOS>`, style markers, task prefixes) 
could be introduced for fine-tuning on specific tasks.

### Features for AlexandriaTokenizer 1.0
A first implementation for my custom BPE tokenizer implements the BPE algorithm using naive approaches that fulfill the expectations on tokenization of small corpuses. However, it cannot handle efficiently the intended production set for this exercise. Hence, I'm documenting the limitations for v1.0 and indicating the next steps for a v2.0

**Limitations**
- Compute Complexity: build_vocab -> O(max_merges)
get_pairs -> O(max(text))
update_token_in_corpus -> O(n*m)
tokenize -> O(n*m*max(merges))
- As the tokenizer builds the vocabulary based only in Spanish text from Wikipedia, it is expected that it will work with worse performance on tokenization of text sources aside from this one, such as coding files, informal conversations, etc. This behavior is expected and it is not intended to be mitigated.
- The tokenization performance in other languages apart from Spanish is expected to be low, as the compression rates would be smaller and posterior language generation process will be more expensive. Alexandria's Tokenizer will not be enhanced to learn vocabulary from other languages. This behavior will not be mitigated in future versions.

## Preprocessing

1. Normalize white spaces in the text
2. Run the tokenization with AlexandriaTokenizer

Document the limitations on the tokenizer (and my thoughts on specialized ones for coding vs a generalist one)
Algorithm Complexity
Mejoras en la tasa de compresión del tokenizador nos permiten aprovechar mejor la ventana de contexto del modelo, más info con menos tokens