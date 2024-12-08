import sklearn.metrics
import numpy as np
import pandas as pd


# Confusion matrix
def confusion_matrix(Y_pred, Y_test, labels=[0, 1, 2]):
    return sklearn.metrics.confusion_matrix(Y_test, Y_pred, labels=labels)


# True Positive (per class)
def tp(Y_pred, Y_test, label) -> int:
    conf_mat = confusion_matrix(Y_pred, Y_test)
    return conf_mat[label, label]  # True Positives for the given class


# False Positive (per class)
def fp(Y_pred, Y_test, label) -> int:
    conf_mat = confusion_matrix(Y_pred, Y_test)
    return np.sum(conf_mat[:, label]) - conf_mat[label, label]  # False Positives for the given class


# False Negative (per class)
def fn(Y_pred, Y_test, label) -> int:
    conf_mat = confusion_matrix(Y_pred, Y_test)
    return np.sum(conf_mat[label, :]) - conf_mat[label, label]  # False Negatives for the given class


# True Negative (per class)
def tn(Y_pred, Y_test, label) -> int:
    conf_mat = confusion_matrix(Y_pred, Y_test)
    total = np.sum(conf_mat)
    class_total = np.sum(conf_mat[label, :]) + np.sum(conf_mat[:, label]) - conf_mat[label, label]
    return total - class_total  # True Negatives for the given class


# Overall Accuracy
def accuracy(Y_pred, Y_test) -> float:
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)
    correct = np.sum(Y_pred == Y_test)
    total = len(Y_test)
    return correct / total if total > 0 else 0


# Overall Precision
def overall_precision(Y_pred, Y_test, labels=[0, 1, 2]) -> float:
    total_tp = sum(tp(Y_pred, Y_test, label) for label in labels)
    total_fp = sum(fp(Y_pred, Y_test, label) for label in labels)
    return total_tp / max(1, total_tp + total_fp)


# Overall Recall
def overall_recall(Y_pred, Y_test, labels=[0, 1, 2]) -> float:
    total_tp = sum(tp(Y_pred, Y_test, label) for label in labels)
    total_fn = sum(fn(Y_pred, Y_test, label) for label in labels)
    return total_tp / max(1, total_tp + total_fn)


# Print all metrics for each class and overall metrics
def print_metrics(Y_pred, Y_test, labels=[0, 1, 2]):
    # Print overall metrics
    print(f"Overall Accuracy: {accuracy(Y_pred, Y_test):.4f}")
    print(f"Overall Precision: {overall_precision(Y_pred, Y_test, labels):.4f}")
    print(f"Overall Recall: {overall_recall(Y_pred, Y_test, labels):.4f}\n")

    # Class-wise metrics
    metrics = pd.DataFrame(columns=["Win", "Draw", "Loss"], index=["TP", "FP", "FN", "TN"])

    for label, friendly_name in zip(labels, ["Win", "Draw", "Loss"]):
        metrics[friendly_name] = [
            tp(Y_pred, Y_test, label),
            fp(Y_pred, Y_test, label),
            fn(Y_pred, Y_test, label),
            tn(Y_pred, Y_test, label),
        ]

    print("Metrics for each class:\n")
    print(metrics.round(4))
