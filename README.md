# Synthetic data in LLMs
Bachelor thesis reearch lab by William Plancke

## Architecture
A custom pipeline is used tie the modules together. Several modules have different options in terms of layers each with passable hyperparameters so there are a lot of different combinations in which the pipeline can be ran.
### Tokenizer
Handmade word tokenization, goes from string of text to a list of token ids, in both directions. There are also functions to save and load vocabularies. The vocabulary can be frozen both with a token limit and manually.
A more advanced option also available is subword BPE tokenization. This tokenizer works very similarly to the word-tokenizer but captures individual characters and maps them to the words it sees in the training corpus, then it merges characters that frequently neighbour each other. As a result the vocabulary contains both frequently occuring words and a wide variety of "word parts" that it can use to partially infer the meaning of unknown words. Longer CPU time but more robust to unseen tokens.
### Embedding
Built-in: torch.nn.Embedding. Assigns each token-id to a latent representation of the meaning of the word.
### Classifier
There's both a normal classifier and a transformer-classifier with their own pipeline:
- Classifier: Embedding -> masked mean pooling -> linear intent classifier.
- Transformer-classifier: Learned positions -> bidirectional encoder -> masked mean -> intent logits.
The latter is more complex and computationally intensive but tokens bring in contextual information through the attention heads and will thus perform better if the input data contains information where this is important (implicit requests and expressions).
## Data
| Dataset  | Scope    | Num. classes | Num. real samples (total) | Example input                               | Example output      |
|----------|----------|--------------|---------------------------|---------------------------------------------|---------------------|
| Clinc150 | Intent   | 151          | 23800                     | can i make a reservation for redrobin       | accept_reservations |
| SST-2    | Sentiment| 2            | 70042                     | hide new secretions from the parental units | negative            |
| AG-News  | Subject  | 4            | 127600                    | Wall St. Bears Claw Back Into the ...       | business            |

### Synthetic data generation
Currently Qwen2.5:3b and Qwen3.5:9b are used to generate synthetic data to be compared to the real world counterpart. Ollama gets used for local inference. The current synthetic data generation proces looks like this:
- Use real training data to establish class mapping, split boundaries, target counts and extact-overlap exclusions
- Prompt the LLM to generate x amount of examples (5-20) giving it a variety of explicit instructions on the format and content.
- Mechanically filter the output examples to check for duplicates or invalid samples (too long, empty strings...)
- Save the dataset along with the parameters/settings it was generated with (seed, llm temperature, batch_size...)
Every time a valid batch is generated it gets saved to the dataset so dataset generation can be paused/resumed.

## Training
Training is handled by `Training/train.py`; experiment entry points currently use CLINC150 and share hyperparameters in `Experiments/common.py`. Run from the repository root after preparing the required datasets:

```bash
python -m Experiments.Real_data
python -m Experiments.Synthetic_data_raw
python -m Experiments.Synthetic_data_with_definition
python -m Experiments.Synthetic_data_finetuned
```

- **Conditions:** train on real data, raw synthetic data, or definition-guided synthetic data. Synthetic foundation datasets must be unique and match the real training set's per-class counts. All conditions use the same real validation split; test data is not loaded.
- **Preparation:** fit the word or BPE tokenizer on training text only, freeze its vocabulary, then encode both splits with fixed-length padding/truncation and attention masks.
- **Optimization:** train the selected classifier and embeddings using cross-entropy loss and SGD, updating once per example (batch size 1). Training rows are shuffled each epoch using a reproducible seed. Current shared settings use a transformer encoder, BPE, 64-dimensional embeddings, 20 epochs, and learning rate 0.01.
- **Fine-tuning:** first train on definition-guided data, then update all model weights for 3 additional epochs at one tenth of the initial learning rate. Replay keeps every original training row and oversamples the targeted supplement to approximately 20% of the mixture. The saved tokenizer and label mapping remain frozen; the optimizer is reset. Supplement labels are not independently verified.
- **Evaluation and saving:** report validation accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix. Each run saves `model.pt`, `tokenizer.json`, `config.json`, and `metrics.json` in a new directory under `Runs/`.

The trainer defaults to CUDA when available, otherwise CPU; the definition-guided experiment entry point currently requires CUDA. AG-News and SST-2 can use the same trainer with `text`/`label` frames and dataset-specific splits/settings, but are not selected by the current experiment entry points.
