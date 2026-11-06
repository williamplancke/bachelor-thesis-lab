import json
from pathlib import Path
PUNCTUATION = [".", "?", "!", ",", ";", ":", "/", "-", "*", "+"]
SENTENCE_ENDER = [".", "?", "!"]
WORD_BOUNDARY = "_"
class SubwordBPETokenizer:
    def __init__(self, lowercase: bool = True, vocabulary_limit: int = 300):
        if vocabulary_limit < 2:
            raise ValueError("Vocabulary limit must accommodate <PAD> and <UNK>.")
        self.lowercase = lowercase
        self.vocabulary_limit = vocabulary_limit
        self.word_frequency_table = {}
        self.word_symbols: dict[str, list[str]] = {}
        self.pair_frequency_table = {}
        self.vocabulary = []
        self.token_to_id = {}
        self.merges = []
        return
    def add_token(self, token):
        if type(token) == tuple:
            left, right = token
            token = left + right
        if token in self.token_to_id:
            return
        if len(self.vocabulary) >= self.vocabulary_limit:
            raise ValueError("Vocabulary limit reached.")
        self.token_to_id[token] = len(self.vocabulary)
        self.vocabulary.append(token)
        return
    def build_baseline_vocabulary(self):
        # Check capacity before adding anything: never silently drop characters.
        alphabet = sorted(set("".join(self.word_symbols)))
        required_tokens = set(alphabet) | {"<PAD>", "<UNK>"} | set(self.vocabulary)
        if len(required_tokens) > self.vocabulary_limit:
            raise ValueError("Vocabulary limit is smaller than the initial alphabet plus special tokens.")
        self.add_token("<PAD>")
        self.add_token("<UNK>")
        for character in alphabet:
            self.add_token(character)
        return
    def fit(self):
        self.build_baseline_vocabulary()
        self.pair_frequency_table = {}
        while len(self.vocabulary) < self.vocabulary_limit:
            self.pair_frequency_table = {}
            for key, word in self.word_symbols.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    if pair in self.pair_frequency_table.keys():
                        self.pair_frequency_table[pair] += 1 * self.word_frequency_table[key]
                    else:
                        self.pair_frequency_table[pair] = 1 * self.word_frequency_table[key]
            if len(self.pair_frequency_table.keys()) == 0:
                break
            selected_pair = self.select_best_pair()
            self.merge(selected_pair)
            self.add_token(selected_pair)
            self.merges.append(selected_pair)
        self.pair_frequency_table = {}
        return
    def select_best_pair(self):
        highest_value = max(self.pair_frequency_table.values())
        highest_keys = [key for key, value in self.pair_frequency_table.items() if value == highest_value]
        sorted_highest_keys = sorted(highest_keys)
        return sorted_highest_keys[0]
    def obtain_token(self, token):
        return self.token_to_id[token]
    def prepare_input(self, text):
        return self.sentences_to_words(text)
    def freeze_vocabulary(self):
        return
    @staticmethod
    def merge_symbols(symbols, selected_pair):
        """Apply one merge left to right without overlapping replacements."""
        left, right = selected_pair
        merged_symbols = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                merged_symbols.append(left + right)
                i += 2
            else:
                merged_symbols.append(symbols[i])
                i += 1
        return merged_symbols

    def merge(self, selected_pair):
        for key, symbols in self.word_symbols.items():
            self.word_symbols[key] = self.merge_symbols(symbols, selected_pair)
    def sentences_to_words(self, text):
        """Normalize text and prefix each word/punctuation unit with the boundary marker."""
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        space_padded = text.lower() if self.lowercase else text
        for char in PUNCTUATION:
            space_padded = space_padded.replace(char, " " + char + " ")
        tokens = space_padded.split()
        cleaned_tokens = remove_empty_tokens(tokens)
        words_with_boundary = []
        for word in cleaned_tokens:
            words_with_boundary.append(WORD_BOUNDARY + word)
        return words_with_boundary
    def words_to_characters(self, text):
        """Collect training counts from the output of sentences_to_words()."""
        characters = []
        for word in text:
            if word in self.word_frequency_table.keys():
                self.word_frequency_table[word] += 1
            else:
                self.word_symbols[word] = list(word)
                self.word_frequency_table[word] = 1
            characters.extend(word)
        # fit() adds the complete, sorted alphabet before learning any merges.
        return characters
    def encode(self, text: str | list[str]) -> list[int]:
        """Encode raw text or boundary-marked words from sentences_to_words()."""
        if "<UNK>" not in self.token_to_id:
            raise ValueError("Fit or load the tokenizer before encoding.")
        if isinstance(text, str):
            words = self.sentences_to_words(text)
        elif isinstance(text, list) and all(isinstance(word, str) for word in text):
            words = text
        else:
            raise TypeError("Expected a string or a list of boundary-marked words.")
        token_ids = []
        unknown_id = self.token_to_id["<UNK>"]
        for word in words:
            symbols = list(word)
            for pair in self.merges:
                symbols = self.merge_symbols(symbols, pair)
            token_ids.extend(self.token_to_id.get(symbol, unknown_id) for symbol in symbols)
        return token_ids

    def decode(self, token_ids: list[int]) -> list[str]:
        """Recover normalized words, skipping padding and preserving unknown markers."""
        symbols = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < len(self.vocabulary):
                raise ValueError(f"Invalid token ID: {token_id}")
            symbol = self.vocabulary[token_id]
            if symbol != "<PAD>":
                symbols.append(symbol)
        return [word for word in "".join(symbols).split(WORD_BOUNDARY) if word]
    def construct_output(self, wordlist: list[str]) -> str:
        capitalise = True
        output = ""
        for word in wordlist:
            if capitalise:
                word = word.capitalize()
                capitalise = False
            if word in SENTENCE_ENDER:
                capitalise = True
            if word in PUNCTUATION:
                output = output.rstrip(" ")
            output += word + " "
        return output
    def save(self, path):
        config = {
            "type": "subword_bpe",
            "format_version": 1,
            "lowercase": self.lowercase,
            "vocabulary_limit": self.vocabulary_limit,
            "word_boundary": WORD_BOUNDARY,
            "vocabulary": self.vocabulary,
            "merges": self.merges,
        }
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
        return

    def load(self, path):
        """Restore this instance without requiring the original training corpus."""
        with Path(path).open("r", encoding="utf-8") as file:
            config = json.load(file)
        if config.get("type") != "subword_bpe" or config.get("format_version") != 1:
            raise ValueError("Unsupported tokenizer file format.")
        if config["word_boundary"] != WORD_BOUNDARY:
            raise ValueError("Tokenizer word-boundary marker does not match.")
        vocabulary = config["vocabulary"]
        if vocabulary[:2] != ["<PAD>", "<UNK>"] or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Invalid tokenizer vocabulary.")
        if len(vocabulary) > config["vocabulary_limit"]:
            raise ValueError("Saved vocabulary exceeds its configured limit.")
        self.lowercase = config["lowercase"]
        self.vocabulary_limit = config["vocabulary_limit"]
        self.vocabulary = vocabulary
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self.merges = [tuple(pair) for pair in config["merges"]]
        self.word_frequency_table = {}
        self.word_symbols = {}
        self.pair_frequency_table = {}
        return
class ByteLevelBPETokenizer:
    def __init__(self):
        return
    def fit(self):
        return
    def prepare_input(text):
        return
    def encode(self):
        return
    def decode(self, token_ids):
        return
    def save(self, path):
        return
    def load(self, path):
        return
def remove_empty_tokens(tokens):
    return list(filter(None, tokens))