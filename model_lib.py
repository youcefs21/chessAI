import sklearn
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from chess import pgn

from typing import Iterable

import feature_handlers as fh

# Use a GPU if available, to speed things up
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# class MoveDataset(Dataset):
#     def __init__(self, moves: pd.DataFrame):
#         moves["Board State"] = moves.apply(lambda row: fh.board_fen_to_image(row["Board State"]), axis=1)
#         self.moves = moves.to_numpy()

#     def __len__(self) -> int:
#         return len(self.moves)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         return self.moves[idx]


def moves_to_numpy(moves: pd.DataFrame):
    moves["Board State"] = moves.apply(lambda row: fh.board_fen_to_image(row["Board State"]), axis=1)
    return moves.to_numpy()


class ChessDataset(Dataset):
    def __init__(self, game_data: pd.DataFrame):
        self.labels = torch.tensor(game_data["Result"].values)
        game_data = game_data.drop(columns=["Result"])
        # print(self.observations["Moves"])
        game_data["Moves"] = game_data.apply(lambda row: moves_to_numpy(row["Moves"]), axis=1)
        # game_data["Moves"] = game_data.apply(lambda row: MoveDataset(row["Moves"]), axis=1)
        self.observations = game_data.to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.labels[idx]


# TESTing dataset
# game_data = fh.pgn_file_to_dataframe("Data/2024-08/xaa.pgn")
# # print(game_data)
# print(game_data.iloc[0]["Moves"]["Board State"])
# dataset = ChessDataset(game_data)
# print(dataset[0][0]["Moves"][0]["Board State"])


class ChessNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Define layers
        # eventually we want output to be for 3 classes - get the probability of  1, -1, or 0 (win, loss, draw)
        # 12 channels, 8x8 board
        cnn_input_dim = [12, 8, 8]
        cnn_output_dim = [16, 3, 3]
        metadata_per_move = len(fh.MOVE_HEADER_NAMES) - 1  # minus 1 since board state is part of cnn

        self.board_cnn = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3),  # 8-3+1=6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 6-2+1=3
            # nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3),  # 3-3+1=1
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(16 * 3 * 3 + metadata_per_move, 128)

        self.move_layer = nn.Sequential()

    def forward(self, x):
        print(x)

        # Define a view to shape the data input
        # Define the forward pass through layers


def test_loss(model: nn.Module, test_loader: DataLoader, loss_function: nn.modules.loss._Loss) -> float:
    """Get the current loss of the data in the test_loader"""

    # Set model to evaluation mode
    model.eval()
    test_loss = 0

    # No backpropagation calculations needed
    with torch.no_grad():
        for data, target in test_loader:

            # Move data to GPU if applicable
            data, target = data.to(device), target.to(device)

            # Predict the data, get the loss based on the prediction
            output = model(data)
            loss = loss_function(output, target)
            test_loss += loss.item()

    # Average calculated loss so training and testing losses can be compared without considering dataset size
    return test_loss / len(test_loader)


def predict(model: nn.Module, test_loader: DataLoader) -> list[int]:
    """Predict the labels of the data in the test_loader using the model"""

    # Set model to evaluation mode
    model.eval()
    predictions = []

    # No backpropagation calculations needed
    with torch.no_grad():
        for data, target in test_loader:

            # Move data to GPU if applicable
            data, target = data.to(device), target.to(device)

            # Predict the data
            output = model(data)
            # Index of the highest value in the predicted tensor will be our chosen class
            # TODO may need to also give the probability of the prediction for each of the 3 classes
            _, predicted = torch.max(output.data, 1)
            predictions += predicted.tolist()

    return predictions


def train(
    model: nn.Module,
    loss_function: nn.modules.loss._Loss,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epoch: int,
    learning_rate: float,
    print_every: int = 100,
) -> tuple[list[float], list[float]]:
    """Train a nn-based model with stochastic gradient descent.

    Returns the training and testing losses for each epoch."""

    # Record the loss for graphing
    train_losses: list[float] = []
    test_losses = []

    # Select the loss function and optimizer
    loss_function = loss_function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(epoch):
        print(f"Starting epoch {e}")

        # Set model to training mode
        model.train()

        # Cumulative loss across all batches for the current epoch (display purposes only)
        current_epoch_train_loss = 0

        for i, (data, target) in enumerate(train_loader):
            # Move data to GPU if applicable
            data, target = data.to(device), target.to(device)

            # Zero the gradients to prevent interference from previous iterations
            optimizer.zero_grad()

            # Predict the data
            output = model(data)
            # Calculate the loss
            loss = loss_function(output, target)
            # Backpropagate to update parameters
            loss.backward()
            optimizer.step()

            current_epoch_train_loss += loss.item()

            # For debug
            if i % print_every == 0:
                print(f"Iter {i}, Mid-training loss: {loss.item()}")

        # Save average train and test loss for later graphing
        train_losses.append(current_epoch_train_loss / len(train_loader))
        test_losses.append(test_loss(model, test_loader, loss_function=loss_function))

        # For debug
        print(f"Epoch {e} training loss: {train_losses[-1]}")
        print(f"Epoch {e} testing loss: {test_losses[-1]}")

    return train_losses, test_losses


def plot_eval_results(train_losses=[], test_losses=[]):
    """Plot the training and testing losses for each epoch"""

    # Plot the data as two lines
    plt.plot(train_losses, "b-", label="Training loss")
    plt.plot(test_losses, "r-", label="Testing loss")
    # Axis titles
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.show()
