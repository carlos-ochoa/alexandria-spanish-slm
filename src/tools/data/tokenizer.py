"""Definition for the Alexandria Tokenizer. Custom implementation of BPE
"""

from collections import Counter
from typing import Tuple, List, Dict
from tqdm import tqdm
import time
import json


class AlexandriaTokenizer:

    def __init__(self, max_merges : int = 8000, load_tokenizer : bool = False):
        self.vocab = {}
        self.vocab_str = {}
        self.stats = {}
        self.max_merges = max_merges
        self.merges = {}
        self.vocab_size = max_merges + 258
        self.most_frequent_merges = Counter()
        if load_tokenizer:
            self.load_tokenizer()
        else:
            self.init_vocab()

    def save_tokenizer(self) -> None:
        """Saves the tokenizer vocab and other metadata to basic json structure
        """
        save_format = {
            "max_merges": self.max_merges,
            "vocab": self.vocab,
            "vocab_str": self.vocab_str,
            "merges": self.merges,
        }
        with open("assets/tokenizer.json", "w") as f:
            json.dump(save_format, f)

    def load_tokenizer(self) -> None:
        """Loads vocab and metadata from previous execution
        """
        with open("assets/tokenizer.json", "r") as f:
            load_format = json.load(f)
        self.vocab = load_format["vocab"]
        self.vocab_str = load_format["vocab_str"]
        self.max_merges = load_format["max_merges"]
        self.merges = load_format["merges"]

    def init_vocab(self) -> None:
        """Inits the first 258 tokens of the vocabulary
        """
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

    def _text_to_bytes(self, corpus : List[str]) -> List[int]:
        """Converts the strings of text into bytes with utf-8 codec

        Args:
            corpus (List[str]): List of texts

        Returns:
            List[int]: Lists of bytes representations
        """
        bytes_corpus = []
        for text in corpus:
            bytes_corpus.append(list(bytes(text, 'utf-8')))
        return bytes_corpus

    def _get_pairs(self, bytes_text : List[int]) -> List[Tuple[int, int]]:
        """Gets all the possible pairs of bytes in the text

        Args:
            bytes_text (List[int]): List of bytes representations

        Returns:
            List[Tuple[int, int]]: List with all possible pairs in a text
        """
        pairs = []
        i = 0
        while i < len(bytes_text)-1:
            pairs.append((bytes_text[i], bytes_text[i+1]))
            i += 1
        return pairs

    def _update_vocab(self, pair : Tuple[int], token_id : int) -> None:
        """Updates the vocabulary and other metadata with new tokens

        Args:
            pair (Tuple[int]): The merge to add to the vocab
            token_id (int): The new ID for that merge
        """
        new_token = self.vocab[pair[0][0]] + self.vocab[pair[0][1]]
        self.vocab[token_id] = new_token
        self.vocab_str[token_id] = self.vocab[token_id].decode('utf-8', errors='replace')
        self.merges[pair[0]] = token_id
        self.most_frequent_merges[pair] = pair[1]

    def _update_token_in_corpus(self, corpus: List[List[int]], pair: Tuple[int, int],
                         new_token: int) -> List[List[int]]:
        """Replaces the merge with its new ID in the corpus

        Args:
            corpus (List[List[int]]): The corpus
            pair (Tuple[int, int]): The merge to replace
            new_token (int): The new token ID

        Returns:
            List[List[int]]: Updated corpus
        """
        new_corpus = []
        for text in corpus:
            new_text = []
            i = 0
            while i < len(text):
                if i < len(text) - 1 and text[i] == pair[0] and text[i + 1] == pair[1]:
                    new_text.append(new_token)
                    i += 2
                else:
                    new_text.append(text[i])
                    i += 1
            new_corpus.append(new_text)
        return new_corpus

    def build_vocab(self, corpus : List[str]) -> None:
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
                pairs_in_article = self._get_pairs(text)
                total_pairs.update(pairs_in_article)
            if total_pairs:
                # Usar criterio de desempate explícito: (frecuencia descendente, par ascendente)
                most_common_pair = max(total_pairs.items(), key=lambda x: (x[1], -x[0][0], -x[0][1]))
                self._update_vocab(most_common_pair, current_id)
                corpus = self._update_token_in_corpus(corpus, most_common_pair[0], current_id)
                current_id += 1
                total_merges += 1
            else:
                break
            pbar.update(1)
        self.save_tokenizer()

    def _find_merges(self, text : List[int]) -> Tuple[Tuple, int]:
        """Finds the possible merges appliable to the text. Returns only
            the earliest created one for merging. Used for tokenizing unseen data.

        Args:
            text (List[int]): The byte-form text to tokenize

        Returns:
            Tuple[Tuple, int]: The merge to apply first
        """
        i = 0
        merges = {}
        while i < len(text)-1: # O(m)
            key = (text[i], text[i+1])
            if key in self.merges: # O(1)
                  merges[key] = self.merges[key]
            i += 1
        if merges:
          merge = min(merges.items(), key=lambda x: x[1]) # O(k) Esto podria ser un heap y ya queda mejor
        else:
          merge = None
        return merge

    def _replace_merges_in_text(self, text : List[int], merge : Tuple[Tuple, int]) -> List[int]:
        """Replaces the given merge in all the ocurrences in the byte-encoded text

        Args:
            text (List[int]): The byte-encoded text
            merge (Tuple[Tuple, int]): Merge to apply

        Returns:
            List[int]: Updated text representation
        """
        i = 0
        new_text = []
        while i < len(text): # O(m)
            if i < len(text)-1 and text[i] == merge[0][0] and text[i+1] == merge[0][1]:
                new_text.append(merge[1])
                i += 2
            else:
                new_text.append(text[i])
                i += 1
        return new_text

    def visualize_tokenization(self, tokens : List[int]) -> str:
        """Represents the string-encoded text separating the tokens that conform it

        Args:
            tokens (List[int]): The byte-encoded text

        Returns:
            str: Visual representation
        """
        text = ""
        for token in tokens:
            text = text + "|" + self.vocab_str[token]
        return text

    def _tokenize_str(self, text : str) -> List[int]:
        """Tokenizes a string-encoded text into learned tokens

        Args:
            text (str): Original text

        Returns:
            List[int]: Representation in tokens
        """
        text = self._text_to_bytes([text])[0] # O(m)
        merge = self._find_merges(text) # O(m)
        while merge is not None: # O(mk)
            text = self._replace_merges_in_text(text, merge) # O(m)
            merge = self._find_merges(text) # O(m)
        return text

    def _add_eos_token(self, text : List[int]) -> List[int]:
        """Adds special EOS token to a given text. Used only for training purposes.

        Args:
            text (List[int]): The tokenized text

        Returns:
            List[int]: New text
        """
        return text.append(257)

    def tokenize(self, text : str | List[str]) -> List[int]:
        """Tokenizes a single string or a corpus. Uses learned vocabulary.

        Args:
            text (str | List[str]): The text to tokenize

        Returns:
            List[int]: Tokenized text
        """
        tokenized_text = []
        if len(text) == 0:
            return tokenized_text
        if isinstance(text, str):
            tokenized_text = self._tokenize_str(text)
        elif isinstance(text, List):
            pbar = tqdm(total=len(text))
            for t in text: # O(nmk)
                tokenized_text.append(self._tokenize_str(t))
                pbar.update(1)
        return tokenized_text

    def decode(self, tokenized_text : List[int] | List[List[int]]) -> str | List[str]:
        """Decodes a token-encoded text and returns its string-encoded representation

        Args:
            tokenized_text (List[int] | List[List[int]]): Tokenized text to decode

        Returns:
            str | List[str]: Decoded text
        """
        text = ""
        for token in tokenized_text:
            text = text + self.vocab[token].decode('utf-8')
        return text