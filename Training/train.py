from pathlib import Path

import pandas as pd
import torch

from Architecture.Classifier import IntentClassifier
from Architecture.Tokenizer import Tokenizer


def main():
    # A fixed seed makes the initial weights reproducible.
    torch.manual_seed(42)
    sequence_length = 20
    embedding_dim = 64
    learning_rate = 0.01

    # 1. Read the labeled sentences.
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "Data" / "intents.csv"
    df = pd.read_csv(data_path)

    label_to_id = {
        "card_missing": 0,
        "card_arrival": 1,
    }

    # 2. Build the vocabulary before creating the embedding layer.
    tokenizer = Tokenizer()
    for text in df["text"]:
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
    for _, row in df.iterrows():
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

        print(f"{row['text']} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
