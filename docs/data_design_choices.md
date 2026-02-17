# Tokenizer design choices

This space discuss the design choices for the dataset used during training, preprocessing and tokenization.

---

## Relevant sources
As part of the design process some relevant papers have been analyzed. Even though not all the lessons have been applied they are still an incredible reference for future decisions.

- [Tiny Stories: How small can language models be and still speak coherent English?](https://arxiv.org/pdf/2305.07759)
- [Deepmind's AI Research Foundations : Represent your language data](https://www.skills.google/paths/3135/course_templates/1452)

## Data sources

Alexandria is a small language model specifically trained in Spanish which goal is to generate plausible text with no further capabilities in mind. 

### Version 1

For this project, the initially considered data source takes Wikipedia articles in Spanish. The main source comes from the official [Wikimedia dataset in Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.es?row=0&views%5B%5D=_20231101ab). 

As additional information on how the training data looks like. We've saved considerable time by using this official Wikimedia's dataset since it is built over the dumps from Wikipedia and all the wiki's parsing has been already performed, giving us the raw text to work with.

For the training we are considering 1.8M articles in Spanish, with a knowledge cutoff from November 2023.

### Version 2

Alexandria 1 was designed with a context window of 256 tokens. Hence, some of the articles in the Wikipedia's dataset were discarded if they would have likely surpassed that limit. 
After this filter, around 700k articles were in the window and become potential data to tokenize and train the model.
Chunking larger articles was also considered, however. Given the model size and the expected capabilities, I did not want to introduce paragraphs without prior context to the model, hence, working with smaller pieces of information was preferred.

However, out of these 700k articles, a simple analysis was performed to understand the composition of the information.

It was found that around 56% of the text were focused on administrative grography, insects, results in football matches and train stations. This clearly showed tendencies that might result in bias for the model.
Besides, the biggest problem was that many of these topics follow a formulaic redaction, and also include words in latin (specially in the insects articles), limiting the expresivity of the model.

After consideration I decided to craft my own synthetic data, by following the ideas discussed in [Tiny Stories: How small can language models be and still speak coherent English?](https://arxiv.org/pdf/2305.07759), adapting them to generate data in Spanish, including desirable features for the model, such as simple language, verbs conjugated in different verbal times and specific narratives.

The dataset has been shared as the [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es) dataset in HuggingFace. Further details on data composition are discussed in its data card.

### Biases and ethical considerations

Alexandria is built mainly as a learning project, hence, I'm not performing extensive analysis or bias considerations for the data curation. Even though the biases identified in [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es) are discussed in its data card.
  
**Known Limitations:**
- All the texts in the dataset are small and below 300 tokens. This data might not be suitable for long-context tasks.
- Topical bias: Tiny-Coop-ES contains fables that talk about cooperation. Every language model trained solely in this data will get this bias towards that kind of stories.

These limitations are acknowledged but not addressed in this iteration. Dealing with a well-balanced and unbiased dataset is out of the scope for this project.

## Tokenization

Alexandria uses a custom BPE Tokenizer as default tokenizer. As part of this project, I decided to implement the tokenizer myself. 

### Dataset composition
At first, I decided to use a sample of [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es) to train the tokenizer. Generating AlexandriaTokenizer V1. And a first version of Alexandria was trained using this data. 
A second version of the tokenizer was trained on fresh synthetic data that will be shared as well.
A complete comparison between V1 and V2 will be discussed, as well as their impact on model training.

- **Data size**: For the tokenizer V1 a subsample on 1500 stories taken from [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es) were used for vocabulary training. Tokenizer V2 was trained on 10k stories from tiny-stories-es.
- **Vocabulary size**: 8,259 tokens.

After training both tokenizers were used to tokenize the entire [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es) dataset to train Alexandria Transformer.

### Special Tokens

Alexandria uses a minimal set of special tokens:

- `<EOS>`: Marks the end of each text
- `<UNK>`: Reserved for unknown sequences (rarely used in byte-level BPE)
- `<PAD>`: Used for vector padding during model training

**Design rationale**: 
A minimal token set keeps the model simple and allocates more vocabulary 
space to Spanish language tokens. The model learns text structure 
(paragraphs, sections) implicitly from the natural formatting of 
Wikipedia articles.

**Future considerations**: 
Additional control tokens (e.g., `<BOS>`, style markers, task prefixes) 
could be introduced for fine-tuning on specific tasks.

### Limitations

**Compute Complexity:** The current implementation of the algorithm has lineal time complexity. Building the vocab is $O(nmk)$ and tokenizing a corpus of unseen data also has $O(nmk)$ time complexity. <br> Where 
- **n = number of texts in corpus**
- **m = size of largest document**
- **k = largest number of found merges during tokenization and number of required merges during vocabulary building**.

Given the current implementation, the tokenizer works in acceptable time windows for big datasets taking until 20 minutes of processing on a Mac Mini M4. However, the performance might not be acceptable for some use cases as other alternatives are more optimized. 

As the tokenizer builds the vocabulary based only in Spanish text from [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es), it is expected that it will work with worse performance on tokenization of text sources aside from this one, such as coding files, informal conversations, etc. This behavior is expected and it is not intended to be mitigated.

The tokenization performance in other languages apart from Spanish is expected to be low, as the compression rates would be smaller and posterior language generation process will be more expensive. Alexandria's Tokenizer will not be enhanced to learn vocabulary from other languages. This behavior will not be mitigated in future versions.

### Performance

**Compression ratios, fertility and final sizes**
| Version | Corpus | Char-level | Byte-level | Word fertility | Size of training data for Alexandria Transformer |
|--------|--------|------------|------------|------------|------------|
|V1| 1.5k stories from tiny-coop-es | 6.77x | 6.55x | 0.89x | 9M
|V2| 10k stories from tiny-stories-es | 5.32x | 5.51x | 1.07x | 12M


These numbers are achieved under evaluation on unseen data for the tokenizer from [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es). It is expected that tokenizing text from other sources such as informal chatting, webs, etc. Perform worse even in Spanish. 
It is also expected to have a lower compression rate in other languages.

AlexandriaTokenizer is designed to manage fable-like texts in Spanish, hence it's optimized to be better at compressing this kind of source and provide optimization for the language model's context window on these tasks.

**Performance on other languages**

AlexandriaTokenizer has learned effectively important morphems in Spanish, and even though it is able to tokenize texts in several other languages, the expected tokens usage will rise up. We can see a simple example of tokenizing a phrase with identical meaning in both Spanish and English.

$|Bienven|ido |al |mundo$ -> 4 tokens<br>
$|W|el|com|e |to |the |w|or|l|d$ -> 10 tokens

Each identified token is separated by the $|$ symbol. English requires 2.5x more tokens than Spanish. We can also see that Spanish morphems are easily identified and some words are considered complete tokens. English provides a near-character-level tokenization instead.

### The decision to build the V2 of AlexandriaTokenizer

As seen in the table above, performance on the V1 of the tokenizer behaves differently to V2. And V2 actually helped the model to achieve better results. However, these results are discussed in Alexandria's Model Card.

I want to point out the information that led me to decide on building a V2 version of the vocabulary. 

Alexandria 1 was trained using the V1 tokenizer. With the entirety of [tiny-coop-es](https://huggingface.co/datasets/hetline/tiny-coop-es) dataset (100k stories) tokenized. However, the first clue of a possible underperformance on language modeling can be spotted if we look at the **word fertility** metric. For V1 we see a fertility of 0.89. This shows a strange behavior considering many SOTA tokenizers have a word fertility of ≈1.21. This measure indicates that V1 tokenizer used to require less than a token per word in the original text.
In other words, on average, a token from V1 represented more than only a word. 

This limits the vocabulary expressivity and limits the capability of Alexandria to generalize and generate words out of simpler morphems. 

The most extreme example of this behavior came up when looking at some tokens referring to the moral of the fables. 
Many tokens in the vocabulary contained complete sentences for the moral. E.g.:

- |Moraleja: Ayudar a otros se siente bien|

Remember the symbol | indicates a single token. Hence, complete morals were treated as single tokens. This alerted two behaviors:
- The tokenizer tended to overfit on formulaic structures like these
- The original dataset used to train V1 contained too many examples of the same moral. Lack of diversity caused this behavior.

Also, it limits the capacity of Alexandria to create and express new morals.

To address this situation, I designed another small synthetic dataset called tiny-stories-es, that was also covering small stories, but with a different variety of topics and not being fables, to give the tokenizer a richer source to learn more useful language patterns for Spanish. 

This resulted in a word fertility of 1.07 and reduced compression rates but better Spanish generalization. Producing 3M more tokens on the same data to train Alexandria compared to V1, also improving the model's results. As discussed in the model card.

### Lessons learned

1. The first naive implementation included list-slicing during merge replacement. This led the algorithm to a time complexity of $O(n^2)$. Removing the list-slicing achieved linear time complexity.
2. The compression rate is a good benchmark but it is important to be realistic on the performance. Tokenizers capture the patterns of the language and style they see and varations might degrade the compression performance. For this project, I chose not to work on a generalist tokenizer but a specialized one.
3. The economy of the model begins on the tokenizer. A better tokenizer that captures well language patterns can help the model to levarage its context window as they can process more information with less tokens.
4. The tokenizer is the firs lens of a language model with the data. It might be able to build richer texts if the tokens capture better roots, morphems and allow expresivity with the limited vicabulary they provide.