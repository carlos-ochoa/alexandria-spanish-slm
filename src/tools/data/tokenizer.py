"""Definition for the Alexandria Tokenizer. Custom implementation of BPE
"""

from collections import Counter
from typing import Tuple, List, Dict
from tqdm import tqdm

class AlexandriaTokenizer:

    def __init__(self, max_merges : int = 23742):
        self.vocab = {}
        self.vocab_str = {}
        self.stats = {}
        self.max_merges = max_merges
        self.merges = {}
        self.most_frequent_merges = Counter()
        self.init_vocab()

    def init_vocab(self) -> None:
        for i in range(256):
            self.vocab[i] = bytes([i])
            try:
                self.vocab_str[i] = bytes([i]).decode('ascii')
                if i == 32:
                    self.vocab_str[i] = '·'
                elif i < 32 or i == 127:
                    self.vocab_str[i] = f'<0x{i:02X}>'
            except:
                self.vocab_str[i] = f'<0x{i:02X}>'
        self.vocab[256] = b'<UNK>'
        self.vocab[257] = b'<EOS>'
        self.vocab_str[256] = '<UNK>'
        self.vocab_str[257] = '<EOS>'

    def _text_to_bytes(self, corpus : list) -> list:
        bytes_corpus = []
        for text in corpus:
            bytes_corpus.append(list(bytes(text, 'utf-8')))
        return bytes_corpus

    def get_pairs(self, bytes_text : str) -> dict:
        # Naive implementation
        pairs = []
        i = 0
        while i < len(bytes_text)-1:
            pairs.append((bytes_text[i], bytes_text[i+1]))
            i += 1
        counter_pairs = Counter(pairs)
        return counter_pairs
    
    def update_vocab(self, pair : Tuple[int], token_id : int) -> None:
        new_token = self.vocab[pair[0][0]] + self.vocab[pair[0][1]]
        self.vocab[token_id] = new_token
        self.vocab_str[token_id] = self.vocab[token_id].decode('utf-8')
        self.merges[pair[0]] = token_id
        self.most_frequent_merges[pair] = pair[1]

    def update_token_in_corpus(self, corpus : list, token : Tuple[int], new_token : int) -> list:
        # Naive implementation
        i = 0
        new_text = []
        new_corpus = []
        for text in corpus:
            while i < len(text)-1:
                if token[0] == text[i] and token[1] == text[i+1]:
                    new_text = list(text)[:i] + [new_token] + list(text)[i+2:]
                i += 1
            if new_text:
                new_corpus.append(new_text)
            else:
                new_corpus.append(text)
            i = 0
        return new_corpus
        
    def build_vocab(self, corpus : list) -> None:
        """A function to tokenize the corpus or text passed

        Args:
            corpus (dict): The text data to tokenize. Expected a list of articles.
        """
        pbar = tqdm(total=self.max_merges)
        total_merges = 0
        current_id = 258
        corpus = self._text_to_bytes(corpus)
        while total_merges < self.max_merges:
            total_pairs = Counter()
            for text in corpus:
                pairs_in_article = self.get_pairs(text)
                total_pairs.update(pairs_in_article)
            if total_pairs:
                most_common_pair = total_pairs.most_common(1)[0]
                self.update_vocab(most_common_pair, current_id)
                corpus = self.update_token_in_corpus(corpus, most_common_pair[0], current_id)
                current_id += 1
                total_merges += 1
            else:
                break
            pbar.update(1)

    def find_merges(self, text : List[int]) -> dict:
        i = 0
        merges = {}
        while i < len(text)-1:
            key = (text[i], text[i+1])
            if key in list(self.merges.keys()):
                merges[key] = self.merges[key]
            i += 1
        merges = Counter(merges)
        return merges
    
    def replace_merges_in_text(self, text : List[int], merges : Dict[Tuple[int, int],int]) -> List[int]:
        i = 0
        new_text = []
        for (merge,token) in merges.items():
            while i < len(text)-1:
                key = (text[i], text[i+1])
                if key == merge:
                    new_text = list(text)[:i] + [token] + list(text)[i+2:]
                i += 1
        return new_text

    def visualize_tokenization(self, tokens : List[int]) -> str:
        text = ""
        for token in tokens:
            text = text + "|" + self.vocab_str[token]
        return text
    
    def __tokenize_str(self, text : str) -> List[int]:
        text = self._text_to_bytes([text])[0]
        merges = self.find_merges(text)
        while merges:
            text = self.replace_merges_in_text(text, merges)
            merges = self.find_merges(text)
        return text
    
    def __add_eos_token(self, text : List[int]) -> List[int]:
        # This method is only for model training purposes
        return text.append(257)

    def tokenize(self, text : str | List[str]) -> List[int]:
        tokenized_text = []
        if len(text) == 0:
            return tokenized_text
        if isinstance(text, str):
            tokenized_text = self.__tokenize_str(text)
        elif isinstance(text, List):
            for t in text:
                tokenized_text.append(self.__tokenize_str(t))
        return tokenized_text
    
    def decode(self, tokenized_text : List[int] | List[List[int]]) -> str | List[str]:
        text = ""
        for token in tokenized_text:
            text = text + self.vocab[token].decode('utf-8')
        return text


t = AlexandriaTokenizer(max_merges=10)
#t.build_vocab(
#    [
#        "Este es un texto del ático.",
#        "Aquí estoy probando algún texto más, para probar",
#        "Aunque no encontremos el contexto, aquí está."
#    ]
#)
t.build_vocab(["hola mundo", "hola amigo"])
tokens = t.tokenize(["Este no se trata de un pretexto", "contexto"])
print(tokens)
for tokenized_text in tokens:
    text = t.visualize_tokenization(tokenized_text)
    print(text)
print(t.most_frequent_merges)
print(t.decode(tokens[0]))

# Compression en mi tokenizer

# escribir pruebas unitarias