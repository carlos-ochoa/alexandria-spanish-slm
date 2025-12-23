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

## Preprocessing

1. Normalize white spaces in the text
2. Add special token <EOS> at the end of each article
3. 

