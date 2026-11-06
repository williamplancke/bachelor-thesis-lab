import torch
from torch import nn


def architecture_config(params):
    """Resolve explicit architecture settings, accepting older flat configs too."""
    architecture = params.get("architecture", "mean-pooling")
    if not isinstance(architecture, str) or architecture.strip().lower() != "mean-pooling":
        raise ValueError(f"Unknown architecture: {architecture!r}. Supported: 'mean-pooling'.")
    model_params = dict(params.get("architecture_params", {}))
    if "embedding_dim" in model_params and "embedding_dim" in params:
        if model_params["embedding_dim"] != params["embedding_dim"]:
            raise ValueError("Conflicting embedding_dim in architecture_params and top-level settings.")
    model_params.setdefault("embedding_dim", params.get("embedding_dim", 64))
    return "mean-pooling", model_params


def build_model(architecture, *, vocabulary_size, number_of_classes, padding_id, architecture_params):
    """Construct a supported architecture without changing the baseline class."""
    models = {"mean-pooling": IntentClassifier}
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture!r}.")
    return models[architecture](
        vocabulary_size=vocabulary_size,
        number_of_classes=number_of_classes,
        padding_id=padding_id,
        **architecture_params,
    )


class IntentClassifier(nn.Module):
    """Embedding -> masked mean pooling -> linear intent classifier."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        number_of_classes: int,
        padding_id: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size, # Keep the same
            embedding_dim=embedding_dim, # Hyperparameter
            padding_idx=padding_id, # index of <PAD>
        )
        self.classifier = nn.Linear(embedding_dim, number_of_classes) # Fully connected layer: (input_connections, output_connections) 

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_vectors = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(token_vectors.dtype)

        # Ignore padding and average the remaining token vectors into one
        # fixed-size representation for each sentence.
        summed_vectors = (token_vectors * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        sentence_vectors = summed_vectors / token_counts

        return self.classifier(sentence_vectors)
