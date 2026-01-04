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

- **Data size**: A subsample of 35 articles will be used to build the vocabulary. Random sampling will be used to pick the articles. This sample provides around 300k tokens.
- **Vocabulary size**: 8,258 tokens.

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
A first implementation for my custom BPE tokenizer implements the BPE algorithm using naive approaches that fulfill the expectations on tokenization of small corpuses. 

**Limitations**
- Compute Complexity: The current implementation of the algorithm has lineal time complexity. Building the vocab is $O(nmk)$ and tokenizing a corpus of unseen data also has $O(nmk)$ time complexity. Where **n = number of texts in corpus, m = size of largest document, k = largest number of found merges during tokenization and number of required merges during vocabulary building**.
- Given the current implementation, the tokenizer works in acceptable time windows for big datasets. However, the performance might not be acceptable as other alternatives are more optimized. I'll discuss the numbers below.
- As the tokenizer builds the vocabulary based only in Spanish text from Wikipedia, it is expected that it will work with worse performance on tokenization of text sources aside from this one, such as coding files, informal conversations, etc. This behavior is expected and it is not intended to be mitigated.
- The tokenization performance in other languages apart from Spanish is expected to be low, as the compression rates would be smaller and posterior language generation process will be more expensive. Alexandria's Tokenizer will not be enhanced to learn vocabulary from other languages. This behavior will not be mitigated in future versions.

### Performance

**Compression rate:**
- 4.04 against character-level tokenization
- 4.14 against byte-level tokenization

These numbers are achieved under evaluation on unseen data for the tokenizer from Wikipedia's Spanish articles. It is expected that tokenizing text from other sources such as informal chatting, webs, etc. Perform worse even in Spanish. 
It is also expected to have a lower compression rate in other languages.

AlexandriaTokenizer is designed to manage effectively enciclopedic-style texts in Spanish, hence it's optimized to be better at compressing this kind of source and provide optimization for the language model's context window on these tasks.

**Performance on other languages**

AlexandriaTokenizer has learned effectively important morphems in Spanish, and even though it is able to tokenize texts in several other languages, the expected tokens usage will rise up. We can see a simple example of tokenizing a phrase with identical meaning in both Spanish and English.

$|Bienven|ido |al |mundo$ -> 4 tokens<br>
$|W|el|com|e |to |the |w|or|l|d$ -> 10 tokens

Each identified token is separated by the $|$ symbol. English requires 2.5x more tokens than Spanish. We can also see that Spanish morphems are easily identified and some words are considered complete tokens. English provides a near-character-level tokenization instead.

**Compute performance**

AlexandriaTokenizer generates a new vocabulary from 300k tokens in around 4 minutes using the next hardware specs.

Tokenizing a corpus of 4M tokens (expected for this exercise) takes around 15 minutes. Given the compression rate, I expect to have a final dataset of 1M tokens to train my language model.

### Lessons learned

1. The first naive implementation included list-slicing during merge replacement. This led the algorithm to a time complexity of $O(n^2)$. Removing the list-slicing achieved linear time complexity.
2. The compression rate is a good benchmark but it is important to be realistic on the performance. Tokenizers capture the patterns of the language and style they see and varations might degrade the compression performance. For this project, I chose not to work on a generalist tokenizer but a specialized one.
3. The economy of the model begins on the tokenizer. A better tokenizer that captures well language patterns can help the model to levarage its context window as they can process more information with less tokens.

## Preprocessing

1. Normalize white spaces in the text
2. Run the tokenization with AlexandriaTokenizer

Document the limitations on the tokenizer (and my thoughts on specialized ones for coding vs a generalist one)
Algorithm Complexity
Mejoras en la tasa de compresión del tokenizador nos permiten aprovechar mejor la ventana de contexto del modelo, más info con menos tokens