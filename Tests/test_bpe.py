import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Architecture.Tokenizers.BytePairTokenizer import SubwordBPETokenizer, WORD_BOUNDARY


class SubwordBPETrainingTests(unittest.TestCase):
    def test_special_tokens_and_sorted_alphabet(self):
        tokenizer = SubwordBPETokenizer(vocabulary_limit=5)
        tokenizer.words_to_characters(tokenizer.sentences_to_words("ba"))
        tokenizer.fit()
        self.assertEqual(tokenizer.vocabulary, ["<PAD>", "<UNK>"] + sorted({WORD_BOUNDARY, "a", "b"}))
        for token_id, token in enumerate(tokenizer.vocabulary):
            self.assertEqual(tokenizer.token_to_id[token], token_id)

    def test_duplicate_tokens_keep_their_original_ids(self):
        tokenizer = SubwordBPETokenizer()
        tokenizer.build_baseline_vocabulary()
        tokenizer.add_token("a")
        tokenizer.add_token("a")
        self.assertEqual(tokenizer.vocabulary, ["<PAD>", "<UNK>", "a"])
        self.assertEqual(tokenizer.token_to_id["a"], 2)

    def test_insufficient_alphabet_capacity_raises_before_adding_tokens(self):
        tokenizer = SubwordBPETokenizer(vocabulary_limit=4)
        tokenizer.words_to_characters(tokenizer.sentences_to_words("ab"))
        with self.assertRaisesRegex(ValueError, "initial alphabet"):
            tokenizer.fit()
        self.assertEqual(tokenizer.vocabulary, [])
        with self.assertRaises(ValueError):
            SubwordBPETokenizer(vocabulary_limit=1)

    def test_merge_replaces_non_overlapping_occurrences(self):
        for symbols, pair, expected in [
            (["a", "a", "a", "a"], ("a", "a"), ["aa", "aa"]),
            (["a", "a", "a"], ("a", "a"), ["aa", "a"]),
            (["a", "b", "a", "b"], ("a", "b"), ["ab", "ab"]),
            (["a", "pp", "a", "pp"], ("a", "pp"), ["app", "app"]),
        ]:
            with self.subTest(symbols=symbols):
                tokenizer = SubwordBPETokenizer()
                tokenizer.word_symbols["example"] = symbols
                tokenizer.merge(pair)
                self.assertEqual(tokenizer.word_symbols["example"], expected)

    def test_weighted_pairs_and_stale_counts(self):
        tokenizer = SubwordBPETokenizer(vocabulary_limit=6)
        tokenizer.words_to_characters(tokenizer.sentences_to_words("ab ab a"))
        tokenizer.pair_frequency_table[("obsolete", "pair")] = 999
        tokenizer.fit()
        self.assertEqual(tokenizer.merges, [(WORD_BOUNDARY, "a")])
        self.assertEqual(tokenizer.word_symbols[WORD_BOUNDARY + "ab"], [WORD_BOUNDARY + "a", "b"])
        self.assertEqual(len(tokenizer.vocabulary), 6)

    def test_fit_stops_when_no_pairs_remain(self):
        tokenizer = SubwordBPETokenizer(vocabulary_limit=100)
        tokenizer.words_to_characters(tokenizer.sentences_to_words("aaaa aaaa"))
        tokenizer.fit()
        self.assertEqual(tokenizer.word_symbols[WORD_BOUNDARY + "aaaa"], [WORD_BOUNDARY + "aaaa"])
        self.assertEqual(tokenizer.word_frequency_table[WORD_BOUNDARY + "aaaa"], 2)
        self.assertEqual(len(tokenizer.merges), 3)
        self.assertEqual(tokenizer.pair_frequency_table, {})

    def test_empty_corpus_and_repeated_fit(self):
        tokenizer = SubwordBPETokenizer()
        tokenizer.fit()
        self.assertEqual(tokenizer.vocabulary, ["<PAD>", "<UNK>"])
        tokenizer.words_to_characters(tokenizer.sentences_to_words("ab"))
        tokenizer.fit()
        vocabulary = tokenizer.vocabulary.copy()
        merges = tokenizer.merges.copy()
        tokenizer.fit()
        self.assertEqual(tokenizer.vocabulary, vocabulary)
        self.assertEqual(tokenizer.merges, merges)

    def test_repeated_word_does_not_corrupt_merged_symbols(self):
        tokenizer = SubwordBPETokenizer()
        tokenizer.words_to_characters(tokenizer.sentences_to_words("happy"))
        tokenizer.merge(("p", "p"))
        symbols = tokenizer.word_symbols[WORD_BOUNDARY + "happy"].copy()
        tokenizer.words_to_characters(tokenizer.sentences_to_words("happy"))
        self.assertEqual(tokenizer.word_symbols[WORD_BOUNDARY + "happy"], symbols)
        self.assertEqual(tokenizer.word_frequency_table[WORD_BOUNDARY + "happy"], 2)

    def test_case_and_whitespace(self):
        tokenizer = SubwordBPETokenizer(lowercase=False)
        self.assertEqual(tokenizer.sentences_to_words("Hello\tWorld!\n"), [WORD_BOUNDARY + word for word in ["Hello", "World", "!"]])
        tokenizer = SubwordBPETokenizer()
        self.assertEqual(tokenizer.sentences_to_words("Hello World!"), [WORD_BOUNDARY + word for word in ["hello", "world", "!"]])

    def test_deterministic_tie_breaking(self):
        first = SubwordBPETokenizer()
        second = SubwordBPETokenizer()
        first.words_to_characters(first.sentences_to_words("ab ac"))
        second.words_to_characters(second.sentences_to_words("ac ab"))
        first.fit()
        second.fit()
        self.assertEqual(first.vocabulary, second.vocabulary)
        self.assertEqual(first.merges, second.merges)


class SubwordBPEEncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SubwordBPETokenizer()
        self.tokenizer.words_to_characters(
            self.tokenizer.sentences_to_words("happy customer! café aaaa")
        )
        self.tokenizer.fit()

    def test_training_word_encoding_matches_learned_segmentation(self):
        expected = [
            self.tokenizer.token_to_id[symbol]
            for symbol in self.tokenizer.word_symbols[WORD_BOUNDARY + "happy"]
        ]
        self.assertEqual(self.tokenizer.encode("happy"), expected)

    def test_unseen_word_uses_known_subwords(self):
        # All characters are known, but this combination was not in training.
        ids = self.tokenizer.encode("happyhappy")
        self.assertNotIn(self.tokenizer.token_to_id["<UNK>"], ids)
        self.assertEqual(self.tokenizer.decode(ids), ["happyhappy"])

    def test_unseen_characters_preserve_known_parts(self):
        ids = self.tokenizer.encode("happy🦆")
        self.assertIn(self.tokenizer.token_to_id["<UNK>"], ids)
        self.assertEqual(self.tokenizer.decode(ids), ["happy<UNK>"])

    def test_raw_and_preprocessed_inputs_match(self):
        text = "Happy customer!"
        self.assertEqual(
            self.tokenizer.encode(text),
            self.tokenizer.encode(self.tokenizer.sentences_to_words(text)),
        )
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), ["happy", "customer", "!"])

    def test_empty_input_and_special_tokens(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.encode([]), [])
        self.assertEqual(self.tokenizer.decode([]), [])
        self.assertEqual(self.tokenizer.decode([0]), [])
        self.assertEqual(self.tokenizer.decode([0, 1]), ["<UNK>"])

    def test_invalid_input_and_ids(self):
        for value in (None, 123, [123]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.tokenizer.encode(value)
        for value in (-1, len(self.tokenizer.vocabulary), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.tokenizer.decode([value])
        with self.assertRaises(ValueError):
            SubwordBPETokenizer().encode("hello")

    def test_save_load_preserves_ids_and_config_without_training_words(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = SubwordBPETokenizer(lowercase=False, vocabulary_limit=2)
            loaded.load(path)
        self.assertEqual(loaded.vocabulary, self.tokenizer.vocabulary)
        self.assertEqual(loaded.token_to_id, self.tokenizer.token_to_id)
        self.assertEqual(loaded.merges, self.tokenizer.merges)
        self.assertEqual(loaded.lowercase, self.tokenizer.lowercase)
        self.assertEqual(loaded.vocabulary_limit, self.tokenizer.vocabulary_limit)
        self.assertEqual(loaded.word_symbols, {})
        for text in ("Happy customer!", "happyhappy", "café", "happy🦆", ""):
            with self.subTest(text=text):
                self.assertEqual(loaded.encode(text), self.tokenizer.encode(text))
                self.assertEqual(loaded.decode(loaded.encode(text)), self.tokenizer.decode(self.tokenizer.encode(text)))

    def test_case_sensitive_save_load(self):
        tokenizer = SubwordBPETokenizer(lowercase=False)
        tokenizer.words_to_characters(tokenizer.sentences_to_words("Hello"))
        tokenizer.fit()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = SubwordBPETokenizer()
            loaded.load(path)
        self.assertFalse(loaded.lowercase)
        self.assertEqual(loaded.decode(loaded.encode("Hello")), ["Hello"])

    def test_reject_old_vocabulary_only_format(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(json.dumps(self.tokenizer.token_to_id), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "format"):
                SubwordBPETokenizer().load(path)


if __name__ == "__main__":
    unittest.main()
