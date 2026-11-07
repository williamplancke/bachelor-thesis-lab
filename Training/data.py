"""Shared model-input preparation and in-memory encoding of datasets."""
import torch
from torch.utils.data import TensorDataset


def prepare_input(tokenizer, text, sequence_length, padding_id):
    """Return fixed-length token IDs and mask, without a batch dimension.

    Oversized inputs are rejected, not silently truncated. An empty input has
    only padding and an all-zero mask, as in the original training pipeline.
    The tokenizer must already be fitted/frozen.
    """
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 1:
        raise ValueError("Sequence length must be a positive integer.")
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    ids = tokenizer.encode(tokenizer.prepare_input(text))
    if len(ids) > sequence_length:
        raise ValueError(f"Input ({len(ids)}) exceeds sequence length ({sequence_length}).")
    padding_length = sequence_length - len(ids)
    input_ids = torch.tensor(ids + [padding_id] * padding_length, dtype=torch.long)
    attention_mask = torch.tensor([1] * len(ids) + [0] * padding_length, dtype=torch.long)
    return input_ids, attention_mask


def encode_dataset(frame, tokenizer, label_to_id, sequence_length, padding_id, label_col_name="label", device=None):
    """Encode rows once, preserving row order and the training label mapping.

    TensorDataset contains IDs [N, L], masks [N, L], and targets [N]. N is the
    number of examples, L is sequence_length. No vocabulary learning occurs here.
    """
    if frame.empty:
        raise ValueError("Cannot encode an empty dataset.")
    if not {"text", label_col_name}.issubset(frame.columns):
        raise ValueError(f"Dataset needs 'text' and {label_col_name!r} columns.")
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 1:
        raise ValueError("Sequence length must be a positive integer.")

    # Preallocate to avoid retaining both a list of tensors and a stacked copy.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = torch.empty((len(frame), sequence_length), dtype=torch.long, device=device)
    attention_masks = torch.empty_like(input_ids)
    targets = torch.empty(len(frame), dtype=torch.long, device=device)
    for position, (text, label) in enumerate(zip(frame["text"], frame[label_col_name])):
        if label not in label_to_id:
            raise ValueError(f"Row {position}: label {label!r} is absent from the training label mapping.")
        ids, mask = prepare_input(tokenizer, text, sequence_length, padding_id)
        input_ids[position] = ids
        attention_masks[position] = mask
        targets[position] = label_to_id[label]
    return TensorDataset(input_ids, attention_masks, targets)
