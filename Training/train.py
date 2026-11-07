import json
from pathlib import Path

import pandas as pd
import torch

from Architecture.Classifier import architecture_config, build_model
from Architecture.Tokenizers.WordTokenizer import Tokenizer as WordTokenizer
from Architecture.Tokenizers.BytePairTokenizer import SubwordBPETokenizer
from Training.metrics import classification_metrics, confusion_matrix
from Training.data import encode_dataset
from Training.runs import run_name, staged_run


def main(df_train, df_val, params, run_dir=None, label_col_name = "label"):
    """Train once on df (text/label columns), save the run, and return its path."""
    if df_train.empty:
        raise ValueError("The training dataset must not be empty.")

    # Use the hyperparameters supplied by the caller.
    torch.manual_seed(params["seed"])
    sequence_length = params["sequence_length"]
    architecture, model_params = architecture_config(params)
    learning_rate = params["learning_rate"]

    # Derive class IDs from this dataset rather than hard-coding its labels.
    labels = sorted(df_train[label_col_name].unique().tolist()) # .tolist() is necessary because json.dump() can't deal with numpy dtypes
    label_to_id = {label: index for index, label in enumerate(labels)}

    # Reject existing destinations now, but don't create a folder until saving.
    if run_dir is not None:
        run_dir = Path(run_dir)
        if run_dir.exists():
            raise FileExistsError(f"Run already exists: {run_dir}")

    # 2. Build the vocabulary before creating the embedding layer.
    print("Creating tokenizer")
    tokenizer_params = params.get("tokenizer_params", {})
    tokenizer = get_tokenizer(params.get("tokenizer_instance", "word"), tokenizer_params)
    for text in df_train["text"]:
        words = tokenizer.prepare_input(text)
        if isinstance(tokenizer, SubwordBPETokenizer):
            # BPE needs corpus-wide pair counts before it can encode text.
            tokenizer.words_to_characters(words)
        else:
            # The word tokenizer builds its vocabulary while encoding.
            tokenizer.encode(words)
    print("Building vocabulary")
    tokenizer.fit()
    tokenizer.freeze_vocabulary()
    padding_id = tokenizer.obtain_token("<PAD>")
    # Encode both splits once, before model creation or any optimization steps.
    # Validation uses the frozen training vocabulary and training label mapping.
    train_dataset = encode_dataset(
        df_train, tokenizer, label_to_id, sequence_length, padding_id, label_col_name,
    )
    validation_dataset = encode_dataset(
        df_val, tokenizer, label_to_id, sequence_length, padding_id, label_col_name,
    )
    print("Tokenizer ready")
    # 3. Create the model, loss function, and optimizer.
    classifier = build_model(
        architecture,
        vocabulary_size=len(tokenizer.vocabulary),
        number_of_classes=len(label_to_id),
        padding_id=padding_id,
        architecture_params=model_params,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

    # 4. Make one training pass, updating the model after each sentence.
    classifier.to("cuda")
    print("Model is on the GPU")
    classifier.train() # Enable training mode
    for epoch in range(params["epochs"]):
        total_loss = 0.0
        # Sample row positions with the same pandas shuffle as the old pipeline.
        shuffled_indices = pd.Series(range(len(train_dataset))).sample(
            frac=1,
            random_state=params["seed"] + epoch,
        )
        for index in shuffled_indices:
            input_ids, attention_mask, target = train_dataset[index]
            # Add the batch dimension: [1, sequence_length] and [1].
            optimizer.zero_grad()  # Clear gradients from the previous sentence.
            logits = classifier(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
            loss = loss_function(logits, target.unsqueeze(0))
            loss.backward()       # Calculate gradients; weights are unchanged yet.
            optimizer.step()      # Update the embedding and classifier weights.
            total_loss += loss.item() # Extract tensor content as 1 scaler value
        average_loss = total_loss / len(df_train)
        print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")
    classifier.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, target in validation_dataset:
            logits = classifier(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
            predicted_id = logits.argmax(dim=1).item()
            targets.append(target.item())
            predictions.append(predicted_id)
    # Summarize validation predictions. Class order matches label_to_id.
    matrix = confusion_matrix(targets, predictions, len(labels))
    metrics = classification_metrics(matrix)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))
    print(f"Validation accuracy: {metrics['accuracy']:.3f}")
    print(f"Validation macro F1: {metrics['macro_f1']:.3f}")
    per_class = {
        label: metrics['per_class'][class_id]
        for label, class_id in label_to_id.items()
    }
    print(pd.DataFrame.from_dict(per_class, orient="index"))

    # Store the label order alongside the matrix so it remains interpretable.
    validation_metrics = {
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": per_class,
    }
    # 5. Save everything needed to recreate the trained classifier.
    config = {
        **params,
        "run_format_version": 1,
        "architecture": architecture,
        "architecture_params": model_params,
        "embedding_dim": model_params["embedding_dim"],
        "total_parameters": sum(parameter.numel() for parameter in classifier.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in classifier.parameters() if parameter.requires_grad
        ),
        "labels": labels,
        "tokenizer_instance": "subword-bpe" if isinstance(tokenizer, SubwordBPETokenizer) else "word",
        "tokenizer_params": tokenizer_params,
        "vocabulary_size": len(tokenizer.vocabulary),
        "number_of_classes": len(label_to_id),
        "padding_id": padding_id,
        "label_to_id": label_to_id,
        "optimizer": "SGD",
        "training_examples": len(df_train),
    }
    if run_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        run_dir = project_root / "Runs" / run_name(config)
    # A failed/interrupted save never exposes a partially written run.
    with staged_run(run_dir) as staging_dir:
        with (staging_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(validation_metrics, file, ensure_ascii=False, indent=2)
        torch.save(classifier.state_dict(), staging_dir / "model.pt")
        tokenizer.save(staging_dir / "tokenizer.json")
        with (staging_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"Saved run to {run_dir}")
    return run_dir

def get_tokenizer(variation, params=None):
    """Create a fresh supported tokenizer from its name or class."""
    params = {} if params is None else params
    tokenizers = {"word": WordTokenizer, "subword-bpe": SubwordBPETokenizer}
    if isinstance(variation, str):
        name = variation.strip().lower()
        if name not in tokenizers:
            raise ValueError(f"Unknown tokenizer: {variation!r}. Choose 'word' or 'subword-bpe'.")
        return tokenizers[name](**params)
    if variation is WordTokenizer or variation is SubwordBPETokenizer:
        return variation(**params)
    raise ValueError("Pass a supported tokenizer name or class, not an instance.")
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    df_train = pd.read_csv(project_root / "Data" / "train_intents.csv")
    df_val = pd.read_csv(project_root / "Data" / "val_intents.csv")
    params = {
        "seed": 42,
        "sequence_length": 20,
        "embedding_dim": 64,
        "learning_rate": 0.01,
        "epochs": 20,
    }
    main(df_train, df_val, params)

