PUNCTUATION = [".", "?", "!", ",", ";", ":", "/", "-", "*", "+"]
WORD_BOUNDARY = "_"
class SubwordBPETokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.word_frequency_table = {}
        self.pair_frequency_table = {}
        self.vocabulary = []
        self.token_to_id = {}
        return
    def add_token(self, token):
        self.vocabulary.append(token)
        self.token_to_id[token] = len(self.vocabulary - 1) 
        return
    def build_baseline_vocabulary(self):
        self.add_token("<PAD>")
        self.add_token("<UNK>")
        return
    def fit(self):
        for word in self.word_frequency_table.keys():
            for i in range(len(word) - 1):
                pair = word[i:i+2]
                if pair in self.pair_frequency_table.keys():
                    self.pair_frequency_table[pair] += 1 * self.word_frequency_table[word]
                else:
                    self.pair_frequency_table[pair] = 1 * self.word_frequency_table[word]
        return
    def sentences_to_words(self, text):
        assert(type(text) == str)
        if self.lowercase:
            lowercase = text.lower()
        space_padded = lowercase
        for char in PUNCTUATION:
            space_padded = space_padded.replace(char, " " + char + " ")
        tokens = space_padded.split(" ")
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
                self.word_frequency_table[word] = 1
            for character in word:
                characters.append(character)
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