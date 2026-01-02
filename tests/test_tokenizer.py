"""Unit tests for the AlexandriaTokenizer
"""

import pytest
from collections import Counter
from src.tools.data.tokenizer import AlexandriaTokenizer


@pytest.fixture
def tokenizer():
    """Fixture providing a fresh AlexandriaTokenizer instance."""
    return AlexandriaTokenizer(max_merges=10, load_tokenizer=False)


@pytest.fixture
def trained_tokenizer():
    """Fixture providing a trained AlexandriaTokenizer instance."""
    tok = AlexandriaTokenizer(max_merges=5, load_tokenizer=False)
    tok.build_vocab([
        "hola mundo",
        "hola amigo",
        "mundo hermoso"
    ])
    return tok


class TestAlexandriaTokenizer:

    def test_init_default_max_merges(self):
        """Test tokenizer initialization with default max_merges."""
        tokenizer = AlexandriaTokenizer(load_tokenizer=False)
        assert tokenizer.max_merges == 8000
        assert isinstance(tokenizer.vocab, dict)
        assert isinstance(tokenizer.merges, dict)
        assert isinstance(tokenizer.most_frequent_merges, Counter)

    def test_init_custom_max_merges(self):
        """Test tokenizer initialization with custom max_merges."""
        tokenizer = AlexandriaTokenizer(max_merges=100, load_tokenizer=False)
        assert tokenizer.max_merges == 100

    def test_init_vocab_base_tokens(self, tokenizer):
        """Test that init_vocab creates 256 base byte tokens."""
        assert len(tokenizer.vocab) >= 258  # 256 bytes + UNK + EOS
        # Check first byte
        assert tokenizer.vocab[0] == bytes([0])
        # Check last byte
        assert tokenizer.vocab[255] == bytes([255])

    def test_init_vocab_special_tokens(self, tokenizer):
        """Test that special tokens are correctly initialized."""
        assert tokenizer.vocab[256] == b'<UNK>'
        assert tokenizer.vocab[257] == b'<EOS>'
        assert tokenizer.vocab_str[256] == '<UNK>'
        assert tokenizer.vocab_str[257] == '<EOS>'

    def test_init_vocab_str_space_representation(self, tokenizer):
        """Test that space character (32) is represented as middle dot."""
        assert tokenizer.vocab_str[32] == '·'

    def test_init_vocab_str_control_characters(self, tokenizer):
        """Test that control characters are represented in hex format."""
        # Test control character (e.g., newline)
        assert tokenizer.vocab_str[10] == '<0x0A>'
        # Test DEL character
        assert tokenizer.vocab_str[127] == '<0x7F>'

    def test_text_to_bytes_single_text(self, tokenizer):
        """Test converting single text to bytes."""
        result = tokenizer._text_to_bytes(["hola"])
        assert len(result) == 1
        assert result[0] == [104, 111, 108, 97]  # UTF-8 bytes for "hola"

    def test_text_to_bytes_multiple_texts(self, tokenizer):
        """Test converting multiple texts to bytes."""
        result = tokenizer._text_to_bytes(["hola", "mundo"])
        assert len(result) == 2
        assert result[0] == [104, 111, 108, 97]
        assert result[1] == [109, 117, 110, 100, 111]

    def test_text_to_bytes_empty_text(self, tokenizer):
        """Test converting empty text to bytes."""
        result = tokenizer._text_to_bytes([""])
        assert result == [[]]

    def test_text_to_bytes_unicode(self, tokenizer):
        """Test converting unicode text to bytes."""
        result = tokenizer._text_to_bytes(["ñ"])
        # ñ in UTF-8 is 0xC3 0xB1
        assert result[0] == [195, 177]

    def test_get_pairs_basic(self, tokenizer):
        """Test getting consecutive pairs from byte sequence."""
        bytes_text = [104, 111, 108, 97]  # "hola"
        pairs = tokenizer._get_pairs(bytes_text)
        assert isinstance(pairs, list)
        assert (104, 111) in pairs  # 'h', 'o'
        assert (111, 108) in pairs  # 'o', 'l'
        assert (108, 97) in pairs   # 'l', 'a'
        assert len(pairs) == 3

    def test_get_pairs_repeated(self, tokenizer):
        """Test getting pairs with repetitions."""
        bytes_text = [97, 97, 97]  # "aaa"
        pairs = tokenizer._get_pairs(bytes_text)
        assert pairs.count((97, 97)) == 2  # 'a', 'a' appears twice

    def test_get_pairs_single_byte(self, tokenizer):
        """Test getting pairs from single byte (should return empty)."""
        bytes_text = [104]
        pairs = tokenizer._get_pairs(bytes_text)
        assert len(pairs) == 0

    def test_get_pairs_empty(self, tokenizer):
        """Test getting pairs from empty sequence."""
        bytes_text = []
        pairs = tokenizer._get_pairs(bytes_text)
        assert len(pairs) == 0

    def test_update_vocab(self, tokenizer):
        """Test updating vocabulary with a new merge."""
        # Create a simple pair: (104, 111) = 'h' + 'o'
        pair = ((104, 111), 5)  # pair and its count
        token_id = 258

        tokenizer._update_vocab(pair, token_id)

        assert token_id in tokenizer.vocab
        assert tokenizer.vocab[token_id] == b'ho'
        assert tokenizer.vocab_str[token_id] == 'ho'
        assert tokenizer.merges[(104, 111)] == token_id
        assert tokenizer.most_frequent_merges[pair] == 5

    def test_update_token_in_corpus_single_replacement(self, tokenizer):
        """Test replacing a token pair in corpus."""
        corpus = [[104, 111, 108, 97]]  # "hola"
        token = (104, 111)  # 'h', 'o'
        new_token = 258

        result = tokenizer._update_token_in_corpus(corpus, token, new_token)

        # Should replace (104, 111) with 258
        assert len(result) == 1
        assert result[0] == [258, 108, 97]

    def test_update_token_in_corpus_multiple_occurrences(self, tokenizer):
        """Test replacing multiple occurrences in corpus."""
        corpus = [[104, 111, 104, 111]]  # "hoho"
        token = (104, 111)  # 'h', 'o'
        new_token = 258

        result = tokenizer._update_token_in_corpus(corpus, token, new_token)

        # Should replace all occurrences
        assert len(result) == 1
        assert result[0] == [258, 258]

    def test_update_token_in_corpus_no_match(self, tokenizer):
        """Test update when token doesn't exist in corpus."""
        corpus = [[104, 111, 108, 97]]  # "hola"
        token = (97, 97)  # 'a', 'a'
        new_token = 258

        result = tokenizer._update_token_in_corpus(corpus, token, new_token)

        # Should remain unchanged
        assert result == corpus

    def test_build_vocab_creates_merges(self, tokenizer):
        """Test that build_vocab creates merge rules."""
        corpus = ["hola mundo", "hola amigo"]

        tokenizer.build_vocab(corpus)

        assert len(tokenizer.merges) > 0
        assert len(tokenizer.vocab) > 258  # More than base vocab

    def test_build_vocab_respects_max_merges(self):
        """Test that build_vocab respects max_merges limit."""
        tokenizer = AlexandriaTokenizer(max_merges=3, load_tokenizer=False)
        corpus = ["hola mundo mundo mundo"]

        tokenizer.build_vocab(corpus)

        # Should have exactly max_merges new tokens
        assert len(tokenizer.merges) == 3
        assert len(tokenizer.vocab) == 258 + 3

    def test_find_merges_basic(self, tokenizer):
        """Test finding learned merges in text."""
        # Manually add a merge
        tokenizer.merges[(104, 111)] = 258  # 'h', 'o' -> 258

        text = [104, 111, 108, 97]  # "hola"
        merge = tokenizer._find_merges(text)

        assert merge is not None
        assert merge[0] == (104, 111)
        assert merge[1] == 258

    def test_find_merges_no_matches(self, tokenizer):
        """Test finding merges when none exist."""
        text = [104, 111, 108, 97]  # "hola"
        merge = tokenizer._find_merges(text)

        assert merge is None

    def test_find_merges_multiple(self, tokenizer):
        """Test finding earliest merge when multiple exist."""
        tokenizer.merges[(104, 111)] = 258  # 'h', 'o' - later merge
        tokenizer.merges[(108, 97)] = 259   # 'l', 'a' - even later merge

        text = [104, 111, 108, 97]  # "hola"
        merge = tokenizer._find_merges(text)

        # Should return the earliest merge (lowest token id)
        assert merge is not None
        assert merge[0] == (104, 111)
        assert merge[1] == 258

    def test_replace_merges_in_text(self, tokenizer):
        """Test replacing merges in text."""
        text = [104, 111, 108, 97]  # "hola"
        merge = ((104, 111), 258)

        result = tokenizer._replace_merges_in_text(text, merge)

        assert 258 in result
        assert len(result) < len(text)
        assert result == [258, 108, 97]

    def test_replace_merges_in_text_no_match(self, tokenizer):
        """Test replace when merge doesn't exist in text."""
        text = [104, 111, 108, 97]
        merge = ((97, 97), 258)

        result = tokenizer._replace_merges_in_text(text, merge)

        # Should return text unchanged
        assert result == text

    def test_visualize_tokenization_basic(self, tokenizer):
        """Test visualization of tokens."""
        tokens = [104, 111, 108, 97]  # h, o, l, a

        result = tokenizer.visualize_tokenization(tokens)

        assert isinstance(result, str)
        assert '|' in result
        assert 'h' in result or '104' in result

    def test_visualize_tokenization_special_tokens(self, tokenizer):
        """Test visualization with special tokens."""
        tokens = [256, 257]  # <UNK>, <EOS>

        result = tokenizer.visualize_tokenization(tokens)

        assert '<UNK>' in result
        assert '<EOS>' in result

    def test_visualize_tokenization_space(self, tokenizer):
        """Test visualization with space character."""
        tokens = [32]  # space

        result = tokenizer.visualize_tokenization(tokens)

        assert '·' in result

    def test_tokenize_single_string(self, trained_tokenizer):
        """Test tokenizing a single string."""
        result = trained_tokenizer.tokenize("hola")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(token, int) for token in result)

    def test_tokenize_list_of_strings(self, trained_tokenizer):
        """Test tokenizing a list of strings."""
        result = trained_tokenizer.tokenize(["hola", "mundo"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(tokens, list) for tokens in result)

    def test_tokenize_empty_string(self, trained_tokenizer):
        """Test tokenizing an empty string."""
        result = trained_tokenizer.tokenize("")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_tokenize_unicode(self, trained_tokenizer):
        """Test tokenizing unicode text."""
        result = trained_tokenizer.tokenize("ñoño")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_decode_basic(self, tokenizer):
        """Test decoding tokens back to text."""
        # UTF-8 bytes for "hola"
        tokens = [104, 111, 108, 97]

        result = tokenizer.decode(tokens)

        assert result == "hola"

    def test_decode_with_merged_tokens(self, tokenizer):
        """Test decoding with merged tokens."""
        # Add a merged token
        tokenizer.vocab[258] = b'ho'
        tokens = [258, 108, 97]  # 'ho' + 'l' + 'a'

        result = tokenizer.decode(tokens)

        assert result == "hola"

    def test_decode_special_tokens(self, tokenizer):
        """Test decoding special tokens."""
        tokens = [256, 257]  # <UNK>, <EOS>

        result = tokenizer.decode(tokens)

        assert '<UNK>' in result
        assert '<EOS>' in result

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """Test that encoding and decoding are inverse operations."""
        original_text = "hola mundo"

        tokens = trained_tokenizer.tokenize(original_text)
        decoded_text = trained_tokenizer.decode(tokens)

        assert decoded_text == original_text

    def test_tokenize_multiple_roundtrip(self, trained_tokenizer):
        """Test encoding/decoding multiple texts."""
        original_texts = ["hola", "mundo"]

        tokens_list = trained_tokenizer.tokenize(original_texts)

        for i, tokens in enumerate(tokens_list):
            decoded = trained_tokenizer.decode(tokens)
            assert decoded == original_texts[i]

    def test_vocab_consistency_after_training(self):
        """Test that vocabulary remains consistent after training."""
        tokenizer = AlexandriaTokenizer(max_merges=5, load_tokenizer=False)

        vocab_size_before = len(tokenizer.vocab)

        tokenizer.build_vocab(["test texto"])

        vocab_size_after = len(tokenizer.vocab)

        # Should have added exactly max_merges new tokens
        assert vocab_size_after == vocab_size_before + 5

    def test_merges_are_deterministic(self):
        """Test that the same corpus produces the same merges."""
        corpus = ["hola mundo", "hola amigo"]

        tokenizer1 = AlexandriaTokenizer(max_merges=3, load_tokenizer=False)
        tokenizer1.build_vocab(corpus)

        tokenizer2 = AlexandriaTokenizer(max_merges=3, load_tokenizer=False)
        tokenizer2.build_vocab(corpus)

        assert tokenizer1.merges == tokenizer2.merges
        assert len(tokenizer1.vocab) == len(tokenizer2.vocab)
