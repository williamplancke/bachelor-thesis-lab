import torch
from torch import nn
from pathlib import Path
from Architecture.Tokenizer import Tokenizer
import json


def save_config(path, payload):
    with open(path, 'w') as file:
        json.dump(payload, file, ensure_ascii=False)
    return
def main():
    seed = 42
    torch.manual_seed(seed)
    tokenizer = Tokenizer()
    # Small demonstration corpus; use your synthetic training corpus later.
    texts = [
        "my card is missing",
        "where is my card",
    ]

    # Build the vocabulary before creating the embedding layer.
    for text in texts:
        tokenizer.encode(tokenizer.prepare_input(text))

    tokenizer.freeze_vocabulary()

    pad_id = tokenizer.obtain_token("<PAD>")

    embedding = nn.Embedding(
        num_embeddings=len(tokenizer.vocabulary), # Keep the same so the whole vocab is in it
        embedding_dim=256, # Real hyperparameter for latent representation
        padding_idx=pad_id,
    )

    tokens = tokenizer.prepare_input("my card is missing")
    token_ids = torch.tensor(
        [tokenizer.encode(tokens)],
        dtype=torch.long,
    )

    vectors = embedding(token_ids)
    print(token_ids.shape)  # torch.Size([1, 4])
    print(vectors.shape)    # torch.Size([1, 4, 256])
    print(vectors)


    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "Runs" / f"sequential_abc_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    emb_config = {
        "num_embeddings": embedding.num_embeddings,
        "embedding_dim": embedding.embedding_dim,
        "learning_rate": 0, # placeholder
        "seed": seed
    }
    tokenizer.save(run_dir / "tokenizer.json")
    save_config(run_dir / "config.json", emb_config)
if __name__ == "__main__":
    main()