import torch
from torch import nn


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
