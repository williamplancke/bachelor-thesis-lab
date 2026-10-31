import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from Architecture.Classifier import IntentClassifier
from Architecture.Tokenizer import Tokenizer


def main(df_train, df_val, params, run_dir=None):
    """Train once on df (text/label columns), save the run, and return its path."""
    if df_train.empty:
        raise ValueError("The training dataset must not be empty.")

    # Use the hyperparameters supplied by the caller.
    torch.manual_seed(params["seed"])
    sequence_length = params["sequence_length"]
    embedding_dim = params["embedding_dim"]
    learning_rate = params["learning_rate"]

    # Derive class IDs from this dataset rather than hard-coding its labels.
    labels = sorted(df_train["label"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {class_id: label for label, class_id in label_to_id.items()}

    # Reserve a new folder so existing runs are not overwritten.
    if run_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        run_name = datetime.now().strftime("intent_%Y%m%d_%H%M%S_%f")
        run_dir = project_root / "Runs" / run_name
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    # 2. Build the vocabulary before creating the embedding layer.
    tokenizer = Tokenizer()
    for text in df_train["text"]:
        tokens = tokenizer.prepare_input(text)
        tokenizer.encode(tokens)
    tokenizer.freeze_vocabulary()
    padding_id = tokenizer.obtain_token("<PAD>")

    # 3. Create the model, loss function, and optimizer.
    classifier = IntentClassifier(
        vocabulary_size=len(tokenizer.vocabulary),
        embedding_dim=embedding_dim,
        number_of_classes=len(label_to_id),
        padding_id=padding_id,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

    # 4. Make one training pass, updating the model after each sentence.
    classifier.train() # Enable training mode
    for epoch in range(params["epochs"]):
        total_loss = 0.0
        for _, row in df_train.iterrows():
            tokens = tokenizer.prepare_input(row["text"])
            input_ids = tokenizer.encode(tokens)
            if len(input_ids) > sequence_length:
                raise ValueError("Input exceeds sequence length!")

            # Real tokens have mask value 1; padding positions have value 0.
            attention_mask = [1] * len(input_ids)
            padding_length = sequence_length - len(input_ids)
            input_ids += [padding_id] * padding_length
            attention_mask += [0] * padding_length

            # The outer list adds a batch dimension: [1, sequence_length].
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
            label_id = label_to_id[row["label"]]
            target = torch.tensor([label_id], dtype=torch.long)

            optimizer.zero_grad()  # Clear gradients from the previous sentence.
            logits = classifier(input_tensor, mask_tensor)
            loss = loss_function(logits, target)
            loss.backward()       # Calculate gradients; weights are unchanged yet.
            optimizer.step()      # Update the embedding and classifier weights.
            total_loss += loss.item() # Extract tensor content as 1 scaler value
        average_loss = total_loss / len(df_train)
        print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")
    classifier.eval()

    with torch.no_grad():
        for _, row in df_val.iterrows():
            tokens = tokenizer.prepare_input(row["text"])
            input_ids = tokenizer.encode(tokens)
            if len(input_ids) > sequence_length:
                raise ValueError("Input exceeds sequence length!")

            # Real tokens have mask value 1; padding positions have value 0.
            attention_mask = [1] * len(input_ids)
            padding_length = sequence_length - len(input_ids)
            input_ids += [padding_id] * padding_length
            attention_mask += [0] * padding_length

            # The outer list adds a batch dimension: [1, sequence_length].
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
            label_id = label_to_id[row["label"]]
            target = torch.tensor([label_id], dtype=torch.long)
            logits = classifier(input_tensor, mask_tensor)
            predicted_id = logits.argmax(dim=1).item()
            predicted_label = id_to_label[predicted_id]
            print(f"Sentence: {row["text"]}")
            print(f"Expected label: {row["label"]}")
            print(f"Predicted label: {predicted_label}")
    # 5. Save everything needed to recreate the trained classifier.
    torch.save(classifier.state_dict(), run_dir / "model.pt")
    tokenizer.save(run_dir / "tokenizer.json")
    config = {
        **params,
        "vocabulary_size": len(tokenizer.vocabulary),
        "number_of_classes": len(label_to_id),
        "padding_id": padding_id,
        "label_to_id": label_to_id,
        "optimizer": "SGD",
        "training_examples": len(df_train),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"Saved run to {run_dir}")
    return run_dir


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

