import unittest

from Architecture.Tokenizers.BytePairTokenizer import SubwordBPETokenizer, WORD_BOUNDARY


class SubwordBPETrainingTests(unittest.TestCase):
    def test_special_tokens_and_sorted_alphabet(self):
        tokenizer = SubwordBPETokenizer(vocabulary_limit=5)
        tokenizer.words_to_characters(["ba"])
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
        tokenizer.words_to_characters(["ab"])
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
        tokenizer.words_to_characters(["ab", "ab", "a"])
        tokenizer.pair_frequency_table[("obsolete", "pair")] = 999
        tokenizer.fit()
        self.assertEqual(tokenizer.merges, [(WORD_BOUNDARY, "a")])
        self.assertEqual(tokenizer.word_symbols[WORD_BOUNDARY + "ab"], [WORD_BOUNDARY + "a", "b"])
        self.assertEqual(len(tokenizer.vocabulary), 6)

    def test_fit_stops_when_no_pairs_remain(self):
        tokenizer = SubwordBPETokenizer(vocabulary_limit=100)
        tokenizer.words_to_characters(["aaaa", "aaaa"])
        tokenizer.fit()
        self.assertEqual(tokenizer.word_symbols[WORD_BOUNDARY + "aaaa"], [WORD_BOUNDARY + "aaaa"])
        self.assertEqual(tokenizer.word_frequency_table[WORD_BOUNDARY + "aaaa"], 2)
        self.assertEqual(len(tokenizer.merges), 3)
        self.assertEqual(tokenizer.pair_frequency_table, {})

    def test_empty_corpus_and_repeated_fit(self):
        tokenizer = SubwordBPETokenizer()
        tokenizer.fit()
        self.assertEqual(tokenizer.vocabulary, ["<PAD>", "<UNK>"])
        tokenizer.words_to_characters(["ab"])
        tokenizer.fit()
        vocabulary = tokenizer.vocabulary.copy()
        merges = tokenizer.merges.copy()
        tokenizer.fit()
        self.assertEqual(tokenizer.vocabulary, vocabulary)
        self.assertEqual(tokenizer.merges, merges)

    def test_repeated_word_does_not_corrupt_merged_symbols(self):
        tokenizer = SubwordBPETokenizer()
        tokenizer.words_to_characters(["happy"])
        tokenizer.merge(("p", "p"))
        symbols = tokenizer.word_symbols[WORD_BOUNDARY + "happy"].copy()
        tokenizer.words_to_characters(["happy"])
        self.assertEqual(tokenizer.word_symbols[WORD_BOUNDARY + "happy"], symbols)
        self.assertEqual(tokenizer.word_frequency_table[WORD_BOUNDARY + "happy"], 2)

    def test_case_and_whitespace(self):
        tokenizer = SubwordBPETokenizer(lowercase=False)
        self.assertEqual(tokenizer.sentences_to_words("Hello\tWorld!\n"), ["Hello", "World", "!"])
        tokenizer = SubwordBPETokenizer()
        self.assertEqual(tokenizer.sentences_to_words("Hello World!"), ["hello", "world", "!"])

    def test_deterministic_tie_breaking(self):
        first = SubwordBPETokenizer()
        second = SubwordBPETokenizer()
        first.words_to_characters(["ab", "ac"])
        second.words_to_characters(["ac", "ab"])
        first.fit()
        second.fit()
        self.assertEqual(first.vocabulary, second.vocabulary)
        self.assertEqual(first.merges, second.merges)


if __name__ == "__main__":
    unittest.main()
