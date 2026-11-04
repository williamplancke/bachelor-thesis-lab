PUNCTUATION = [".", "?", "!", ",", ";", ":", "/", "-", "*", "+"]
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
    def merge(self, selected_pair):
        left, right = selected_pair
        for key, word in self.word_symbols.items():
            merged_symbols = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
                    merged_symbols.append(left + right)
                    i += 2
                else:
                    merged_symbols.append(word[i])
                    i += 1
            self.word_symbols[key] = merged_symbols
    def sentences_to_words(self, text):
        assert(type(text) == str)
        space_padded = text.lower() if self.lowercase else text
        for char in PUNCTUATION:
            space_padded = space_padded.replace(char, " " + char + " ")
        tokens = space_padded.split()
        return remove_empty_tokens(tokens)
    def words_to_characters(self, text):
        words_with_boundary = []
        characters = []
        for word in text:
            words_with_boundary.append(WORD_BOUNDARY + word)
        for word in words_with_boundary:
            if word in self.word_frequency_table.keys():
                self.word_frequency_table[word] += 1
            else:
                self.word_symbols[word] = list(word)
                self.word_frequency_table[word] = 1
            characters.extend(word)
        # fit() adds the complete, sorted alphabet before learning any merges.
        return characters
    def encode(self):
        return
    def decode(self):
        return
    def save(self, path):
        return
    def load(self, path):
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