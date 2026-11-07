import torch
from torch import nn


def architecture_config(params):
    """Resolve explicit architecture settings, accepting older flat configs too."""
    architecture = params.get("architecture", "mean-pooling")
    supported = ("mean-pooling", "transformer-encoder")
    if not isinstance(architecture, str) or architecture.strip().lower() not in supported:
        raise ValueError(f"Unknown architecture: {architecture!r}. Supported: {supported}.")
    architecture = architecture.strip().lower()
    model_params = dict(params.get("architecture_params", {}))
    if "embedding_dim" in model_params and "embedding_dim" in params:
        if model_params["embedding_dim"] != params["embedding_dim"]:
            raise ValueError("Conflicting embedding_dim in architecture_params and top-level settings.")
    model_params.setdefault("embedding_dim", params.get("embedding_dim", 64))
    if architecture == "transformer-encoder":
        model_params.setdefault("number_of_heads", 4)
        model_params.setdefault("number_of_layers", 1)
        model_params.setdefault("feedforward_dim", 128)
        model_params.setdefault("dropout", 0.1)
        model_params.setdefault("max_sequence_length", params.get("sequence_length", 128))
    return architecture, model_params


def build_model(architecture, *, vocabulary_size, number_of_classes, padding_id, architecture_params):
    """Construct a supported architecture without changing the baseline class."""
    models = {
        "mean-pooling": IntentClassifier,
        "transformer-encoder": TransformerIntentClassifier,
    }
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


class TransformerIntentClassifier(nn.Module):
    """Learned positions -> bidirectional encoder -> masked mean -> intent logits."""

    def __init__(
        self, vocabulary_size, embedding_dim, number_of_classes, padding_id,
        number_of_heads=4, number_of_layers=1, feedforward_dim=128,
        dropout=0.1, max_sequence_length=128,
    ):
        super().__init__()
        if embedding_dim < 1 or number_of_heads < 1 or embedding_dim % number_of_heads:
            raise ValueError("embedding_dim must be positive and divisible by number_of_heads.")
        if number_of_layers < 1 or feedforward_dim < 1 or max_sequence_length < 1:
            raise ValueError("Layer count, feed-forward size, and maximum sequence length must be positive.")
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=padding_id).to("cuda")
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim).to("cuda")
        layer = nn.TransformerEncoderLayer( # <-- Self attention is here
            d_model=embedding_dim,
            nhead=number_of_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu", # gaussian error linear unit
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=number_of_layers,
            norm=nn.LayerNorm(embedding_dim), enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(embedding_dim, number_of_classes)

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("IDs and attention mask must have matching [batch, sequence] shapes.")
        length = input_ids.shape[1]
        if not 1 <= length <= self.max_sequence_length:
            raise ValueError(f"Input length must be between 1 and {self.max_sequence_length}.")
        positions = torch.arange(length, device=input_ids.device)
        vectors = self.embedding(input_ids) + self.position_embedding(positions)
        real_tokens = attention_mask.bool()
        padding_mask = ~real_tokens
        # Attention cannot safely attend to an entirely masked row. Expose one
        # dummy position, but still exclude it from pooling using the original mask.
        padding_mask = padding_mask.clone()
        padding_mask[~real_tokens.any(dim=1), 0] = False
        vectors = self.encoder(vectors, src_key_padding_mask=padding_mask)
        mask = real_tokens.unsqueeze(-1).to(vectors.dtype)
        pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)
