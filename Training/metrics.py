import numpy as np


def confusion_matrix(targets, predictions, number_of_classes):
    """Count examples by actual class (rows) and predicted class (columns).

    Class IDs must be integers from 0 to number_of_classes - 1.
    """
    if number_of_classes < 1:
        raise ValueError("number_of_classes must be positive.")
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have the same length.")

    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for actual, predicted in zip(targets, predictions):
        for class_id in (actual, predicted):
            if not isinstance(class_id, (int, np.integer)) or not 0 <= class_id < number_of_classes:
                raise ValueError(f"Invalid class ID: {class_id}")
        matrix[actual, predicted] += 1
    return matrix


def classification_metrics(matrix):
    """Calculate metrics from a confusion matrix produced above.

    Undefined precision/recall/F1 values are reported as 0. Macro F1 is
    the unweighted average over all classes, including absent classes.
    """
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Metrics require at least one evaluated example.")

    per_class = []
    for class_id in range(len(matrix)):
        true_positive = int(matrix[class_id, class_id])
        actual_count = int(matrix[class_id, :].sum())
        predicted_count = int(matrix[:, class_id].sum())

        # Precision: of the examples predicted as this class, how many were right?
        precision = true_positive / predicted_count if predicted_count else 0.0
        # Recall: of the actual examples of this class, how many did we find?
        recall = true_positive / actual_count if actual_count else 0.0
        # F1 balances precision and recall using their harmonic mean.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_count,
        })

    return {
        "accuracy": float(matrix.trace() / total),
        "macro_f1": sum(item["f1"] for item in per_class) / len(per_class),
        "per_class": per_class,
    }
