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


# True Positive Rate (Recall per class)
def tpr(Y_pred, Y_test, label) -> float:
    return tp(Y_pred, Y_test, label) / max(1, tp(Y_pred, Y_test, label) + fn(Y_pred, Y_test, label))


# False Positive Rate per class
def fpr(Y_pred, Y_test, label) -> float:
    return fp(Y_pred, Y_test, label) / max(1, fp(Y_pred, Y_test, label) + tn(Y_pred, Y_test, label))


# Overall Accuracy
def accuracy(Y_pred, Y_test) -> float:
    # Ensure the predicted and true labels are numpy arrays (for comparison)
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)

    # Calculate the number of correct predictions
    correct = np.sum(Y_pred == Y_test)
    total = len(Y_test)

    # Return the accuracy as the ratio of correct predictions to total
    return correct / total if total > 0 else 0


# Print all metrics for each class (Win, Loss, Draw)
def print_metrics(Y_pred, Y_test, labels=[0, 1, 2]):
    # Print Accuracy
    print(f"Accuracy: {accuracy(Y_pred, Y_test):.4f}")

    metrics = pd.DataFrame(columns=["Win", "Draw", "Loss"], index=["TP", "FP", "FN", "TN", "TPR", "FPR"])

    # Print metrics for each class
    for label, friendly_name in zip(labels, ["Win", "Draw", "Loss"]):
        metrics[friendly_name] = [
            tp(Y_pred, Y_test, label),
            fp(Y_pred, Y_test, label),
            fn(Y_pred, Y_test, label),
            tn(Y_pred, Y_test, label),
            tpr(Y_pred, Y_test, label),
            fpr(Y_pred, Y_test, label),
        ]
        # print(f"Class {label}:")
        # print(f"  TP: {tp(Y_pred, Y_test, label)}")
        # print(f"  FP: {fp(Y_pred, Y_test, label)}")
        # print(f"  FN: {fn(Y_pred, Y_test, label)}")
        # print(f"  TN: {tn(Y_pred, Y_test, label)}")
        # print(f"  TPR (Recall): {tpr(Y_pred, Y_test, label)}")
        # print(f"  FPR: {fpr(Y_pred, Y_test, label)}")
        # print()
    print(metrics)
