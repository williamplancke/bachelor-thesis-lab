# Synthetic data in LLMs

Bachelor thesis reearch lab by William Plancke

## Architecture
A custom pipeline is used tie the modules together.
### Tokenizer
Handmade word tokenization, goes from string of text to a list of token ids, in both directions. There are also functions to save and load vocabularies. The vocabulary can be frozen both with a token limit and manually.
### Embedding
Built-in: torch.nn.Embedding. Assigns each token-id to a latent representation of the meaning of the word.
### Classifier
