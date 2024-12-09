import numpy as np
import pandas as pd
import torch
import pickle
from training import ChessDataset, ChessNN, collate_fn, pgn_file_to_dataframe, accuracy, predict, get_data_loaders
from torch.utils.data import DataLoader
import sklearn.metrics
import matplotlib.pyplot as plt

# use same device configuration as training
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("mps available")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Confusion matrix
def confusion_matrix(y_prediction, y_test, labels=None):
    if labels is None:
        labels = [0, 1, 2]
    return sklearn.metrics.confusion_matrix(y_test, y_prediction, labels=labels)


# True Positive (per class)
def tp(y_prediction, y_test, label) -> int:
    conf_mat = confusion_matrix(y_prediction, y_test)
    return conf_mat[label, label]  # True Positives for the given class


# False Positive (per class)
def fp(y_prediction, y_test, label) -> int:
    conf_mat = confusion_matrix(y_prediction, y_test)
    return np.sum(conf_mat[:, label]) - conf_mat[label, label]  # False Positives for the given class


# False Negative (per class)
def fn(y_prediction, y_test, label) -> int:
    conf_mat = confusion_matrix(y_prediction, y_test)
    return np.sum(conf_mat[label, :]) - conf_mat[label, label]  # False Negatives for the given class


# True Negative (per class)
def tn(y_prediction, y_test, label) -> int:
    conf_mat = confusion_matrix(y_prediction, y_test)
    total = np.sum(conf_mat)
    class_total = np.sum(conf_mat[label, :]) + np.sum(conf_mat[:, label]) - conf_mat[label, label]
    return total - class_total  # True Negatives for the given class


# Overall Precision
def overall_precision(y_prediction, y_test, labels=[0, 1, 2]) -> float:
    total_tp = sum(tp(y_prediction, y_test, label) for label in labels)
    total_fp = sum(fp(y_prediction, y_test, label) for label in labels)
    return total_tp / max(1, total_tp + total_fp)


# Overall Recall
def overall_recall(y_prediction, y_test, labels=[0, 1, 2]) -> float:
    total_tp = sum(tp(y_prediction, y_test, label) for label in labels)
    total_fn = sum(fn(y_prediction, y_test, label) for label in labels)
    return total_tp / max(1, total_tp + total_fn)


# Print all metrics for each class and overall metrics
def print_metrics(y_prediction, y_test, labels=[0, 1, 2]):
    # Print overall metrics
    print(f"Overall Accuracy: {accuracy(y_prediction, y_test):.4f}")
    print(f"Overall Precision: {overall_precision(y_prediction, y_test, labels):.4f}")
    print(f"Overall Recall: {overall_recall(y_prediction, y_test, labels):.4f}\n")

    # Class-wise metrics
    metrics = pd.DataFrame(columns=["Win", "Draw", "Loss"], index=["TP", "FP", "FN", "TN"])

    for label, friendly_name in zip(labels, ["Win", "Draw", "Loss"]):
        metrics[friendly_name] = [
            tp(y_prediction, y_test, label),
            fp(y_prediction, y_test, label),
            fn(y_prediction, y_test, label),
            tn(y_prediction, y_test, label),
        ]

    print("Metrics for each class:\n")
    print(metrics.round(4))

def plot_metrics_vs_moves(model, train_loader, test_loader, move_limits, game_dataset):
    """Plot accuracy, precision and recall vs number of moves used"""
    train_metrics = {"accuracy": [], "precision": [], "recall": []}
    test_metrics = {"accuracy": [], "precision": [], "recall": []}

    for move_limit in move_limits:
        # update move limit for both loaders
        game_dataset.move_limit = move_limit

        # get predictions for train set
        y_pred_train = predict(model, train_loader)
        y_true_train = []
        for _, _, _, x, _ in train_loader:
            y_true_train.extend(x.tolist())

        # get predictions for test set
        y_pred_test = predict(model, test_loader)
        y_true_test = []
        for _, _, _, x, _ in test_loader:
            y_true_test.extend(x.tolist())

        # calculate metrics for train set
        train_metrics["accuracy"].append(accuracy(y_pred_train, y_true_train))
        train_metrics["precision"].append(overall_precision(y_pred_train, y_true_train))
        train_metrics["recall"].append(overall_recall(y_pred_train, y_true_train))

        # calculate metrics for test set
        test_metrics["accuracy"].append(accuracy(y_pred_test, y_true_test))
        test_metrics["precision"].append(overall_precision(y_pred_test, y_true_test))
        test_metrics["recall"].append(overall_recall(y_pred_test, y_true_test))

    # create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # plot accuracy
    ax1.plot(move_limits, train_metrics["accuracy"], "b-", label="Train")
    ax1.plot(move_limits, test_metrics["accuracy"], "r--", label="Test")
    ax1.set_title("Accuracy vs Moves")
    ax1.set_xlabel("Number of Moves")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # plot precision
    ax2.plot(move_limits, train_metrics["precision"], "b-", label="Train")
    ax2.plot(move_limits, test_metrics["precision"], "r--", label="Test")
    ax2.set_title("Precision vs Moves")
    ax2.set_xlabel("Number of Moves")
    ax2.set_ylabel("Precision")
    ax2.legend()
    ax2.grid(True)

    # plot recall
    ax3.plot(move_limits, train_metrics["recall"], "b-", label="Train")
    ax3.plot(move_limits, test_metrics["recall"], "r--", label="Test")
    ax3.set_title("Recall vs Moves")
    ax3.set_xlabel("Number of Moves")
    ax3.set_ylabel("Recall")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("metrics_vs_moves.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix with labels"""
    cm = confusion_matrix(y_pred, y_true)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    # add labels
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Black Win", "Draw", "White Win"])
    ax.set_yticklabels(["Black Win", "Draw", "White Win"])

    # rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # add colorbar
    plt.colorbar(im)

    # add numbers to cells
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title(title)
    plt.tight_layout()

    return fig


def load_model(path: str) -> ChessNN:
    """Load a saved model"""
    model = ChessNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()  # Set to evaluation mode
    return model


if __name__ == "__main__":
    MODEL_PATH = "submission/model.pt"
    TEST_DATA_PATH = "Data/2024-08/xaa.pgn"

    try:
        # load model and create initial test loader
        model = load_model(MODEL_PATH)
        model.to(device)
        model.eval()

        # load data
        game_data = pgn_file_to_dataframe(TEST_DATA_PATH)
        game_dataset = ChessDataset(game_data, 10)

        # create data loaders
        train_loader, validation_loader, test_loader, train_set_split = get_data_loaders(game_dataset, batch_size=128)

        # plot metrics for different move limits
        print("Running tests and gathering metrics...")
        move_limits = [2, 4, 6, 8, 10, 14, 20, 24, 30]
        plot_metrics_vs_moves(model, train_loader, test_loader, move_limits, game_dataset)

        # generate confusion matrices for a few key move limits
        for moves in [10, 20, 30]:
            game_dataset.move_limit = moves

            # get predictions
            y_pred = predict(model, test_loader)
            y_true = []
            for _, _, _, target, _ in test_loader:
                y_true.extend(target.tolist())

            # plot and save confusion matrix
            fig = plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix ({moves} moves)")
            plt.savefig(f"confusion_matrix_{moves}moves.png")
            plt.close()

            # print detailed metrics
            print(f"\nMetrics for {moves} moves:")
            print_metrics(y_pred, y_true)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model file and test data file exist at the specified paths.")
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
