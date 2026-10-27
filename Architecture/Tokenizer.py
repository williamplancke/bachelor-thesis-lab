class Tokenizer():
    def __init__(self, vocabulary = None, vocabulary_size = -1):
        if vocabulary == None:
            self.vocabulary = []
            self.vocabulary.append("<PAD>")
            self.vocabulary.append("<UNK>")
        else:
            self.vocabulary = list(vocabulary)
            if "<PAD>" not in self.vocabulary:
                self.vocabulary.append("<PAD>")
            if "<UNK>" not in self.vocabulary:
                self.vocabulary.append("<UNK>")
        self.punctuation = [".", "!", "?", ",", ";"]
        self.sentence_end = [".", "!", "?"]
        self.vocabulary_size = vocabulary_size
        self.frozen_vocabulary = False
    def freeze_vocabulary(self, reverse = False):
        if not reverse:
            self.frozen_vocabulary = True
        else:
            self.frozen_vocabulary = False
        return
    def obtain_token(self, input):
        if input not in self.vocabulary:
            if not self.frozen_vocabulary:
                if self.vocabulary_size > 0 and len(self.vocabulary) >= self.vocabulary_size:
                    self.freeze_vocabulary()
                    input = "<UNK>"
                else:
                    self.vocabulary.append(input)
            else:
                input = "<UNK>"
        return self.vocabulary.index(input)
    def obtain_word(self, input):
        if 0 <= input < len(self.vocabulary):
            return self.vocabulary[input]
        else:
            return "<UNK>"
    def encode(self, tokens):
        token_ids = []
        for token in tokens:
            token_ids.append(self.obtain_token(token))
        return token_ids
    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            tokens.append(self.obtain_word(token_id))
        return tokens
    def prepare_input(self, input):
        assert(type(input) == str)
        lowercase = input.lower()
        space_padded = lowercase
        for char in self.punctuation:
            space_padded = space_padded.replace(char, " " + char + " ")
        tokens = space_padded.split()
        return tokens
    def construct_output(self, tokens):
        sentence_start = True
        output = ''
        for token in tokens:
            if sentence_start:
                token = token.capitalize()
                sentence_start = False
            if token in self.punctuation:
                output = output.rstrip(" ")
            if token in self.sentence_end:
                sentence_start = True
            output += token
            output += ' '
        return output
def main():
    text = "Natural language processing is fun. Natural language models use tokens!"
    tok = Tokenizer()
    tokens = tok.prepare_input(text)
    token_ids = tok.encode(tokens)
    output_tokens = tok.decode(token_ids)
    output_text = tok.construct_output(output_tokens)
    print(f"Token list: {tokens}")
    print(f"Vocabulary: {tok.vocabulary}")
    print(f"Encoded id's list: {token_ids}")
    print(f"Decoded text: {output_text}")

if __name__ == '__main__':
    main()