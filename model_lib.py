import sklearn
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler

from chess import pgn

from typing import Iterable

import feature_handlers as fh
import performance_metrics as pm

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


def moves_to_numpy(moves: pd.Series) -> pd.Series:
    moves = moves.fillna(0)
    moves["Board State"] = moves.apply(lambda row: fh.board_fen_to_image(row["Board State"]), axis=1)
    # print(moves["Board State"].iloc[0].shape)
    return moves


class ChessDataset(Dataset):
    def __init__(self, game_data: pd.DataFrame, only_use_first_X_moves: int | None = None):
        game_data = game_data.dropna()
        self.labels = torch.tensor(game_data["Result"].to_numpy())

        game_data["Moves"] = game_data.apply(lambda row: moves_to_numpy(row["Moves"]), axis=1)
        # moves = game_data["Moves"]
        # print(moves.iloc[0].columns.values)
        self.board_states = game_data.apply(lambda row: row["Moves"]["Board State"].to_list(), axis=1)
        # moves = moves.drop(columns=["Board State"])
        # self.moves = moves.to_numpy()
        # print(self.board_states.shape)
        # print(len(self.board_states))
        # print(len(self.board_states[0]))
        # print(len(self.board_states[1]))
        # print(self.board_states.iloc[0].shape)
        # print(self.board_states.iloc[0].dtype)
        self.moves = game_data.apply(lambda row: row["Moves"].drop(columns=["Board State"]).to_numpy(), axis=1).to_numpy()
        # print(self.board_states)
        # print(self.moves)

        # game_data["Moves"] = game_data.apply(lambda row: MoveDataset(row["Moves"]), axis=1)
        self.game_metadata = game_data.drop(columns=["Moves", "Result"]).to_numpy()

        # Check that there are no empty samples
        # valid_observations = [
        #     obs for obs in self.observations if len(obs) > 0
        # ]  # TODO ask muzamil about this? removing this would cause a mismatch in the other parts of the data...
        # print(valid_observations)
        # print("Difference in observations:", len(self.observations) - len(valid_observations))
        # self.observations = np.array(valid_observations)
        # TODO standard scale the observations
        # Technically we should only fit on the training data... but okay for now
        self.game_metadata = StandardScaler().fit_transform(self.game_metadata)

        # TODO ask muzamil about this? removing this would cause a mismatch in the other parts of the data...
        # valid_indices = [i for i, row in enumerate(self.board_states) if len(row) > 0 and len(self.moves[i]) > 0]
        # print("Difference in indices:", len(self.board_states) - len(valid_indices))
        # self.board_states = self.board_states[valid_indices]
        # self.moves = self.moves[valid_indices]

        self.move_limit = only_use_first_X_moves

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert self.board_states[idx] to a numpy array first, then to a tensor
        board_state_tensor = torch.tensor(np.array(self.board_states[idx]))
        moves = self.moves[idx]

        if self.move_limit is not None:
            moves = moves[: self.move_limit]
            board_state_tensor = board_state_tensor[: self.move_limit]

        return self.game_metadata[idx], moves, board_state_tensor, self.labels[idx]


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
        # eventually we want output to be for 3 classes - get the probability of  2, 0, or 1 (win, loss, draw)
        # 12 channels, 8x8 board

        # cnn_input_dim = [12, 8, 8]
        # cnn_output_dim = [16, 3, 3]
        metadata_per_move = len(fh.MOVE_HEADER_NAMES) - 1  # minus 1 since board state is part of cnn
        metadata_per_game = len(fh.HEADERS_TO_KEEP) - 1  # minus 1 since result is a label

        # CNN for board state
        self.board_cnn = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3),  # 8x8 -> 6x6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 6x6 -> 3x3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 3x3 -> 1x1
            nn.ReLU(),
        )

        # Fully connected layer to flatten CNN output
        self.flatten_cnn_output = nn.Linear(64, 128)

        # RNN for move sequences
        self.rnn = nn.LSTM(
            input_size=128 + metadata_per_move,  # CNN features + metadata
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )

        # Fully connected layer for final prediction
        self.fc = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128 + metadata_per_game, 128)
        self.fc3 = nn.Linear(128, 3)  # Output: [Win, Loss, Draw]

    def forward(self, game_metadata, moves, board_states, lengths):
        # Get batch and sequence dimensions
        batch_size, seq_len, _, _, _ = board_states.size()

        # Pass board states through CNN
        cnn_out = self.board_cnn(board_states.view(-1, 12, 8, 8))  # (batch_size * seq_len, cnn_output_dim)
        cnn_out = cnn_out.view(batch_size * seq_len, -1)  # Flatten for the fully connected layer

        # Pass through the fully connected layer to flatten
        cnn_out = self.flatten_cnn_output(cnn_out)  # (batch_size * seq_len, 128)

        # Reshape back to (batch_size, seq_len, -1) for RNN
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 128)

        # Combine CNN output with moves
        combined_features = torch.cat((cnn_out, moves), dim=2)  # (batch_size, seq_len, 128 + metadata_per_move)

        # Pack sequences for RNN
        packed_features = pack_padded_sequence(combined_features, lengths.to("cpu"), batch_first=True, enforce_sorted=False)

        # Debugging output
        # print(f"Packed features shape: {packed_features.data.size()}, LSTM input size: {self.rnn.input_size}")

        # Pass through RNN
        packed_rnn_out, _ = self.rnn(packed_features)

        # Unpack the RNN output
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)  # (batch_size, seq_len, rnn_hidden_dim)

        # Use the last valid RNN output for each sequence
        idx = (lengths - 1).clone().detach().to(device=rnn_out.device).unsqueeze(1).unsqueeze(2).expand(-1, 1, rnn_out.size(2)).long()
        last_rnn_out = rnn_out.gather(1, idx).squeeze(1)  # (batch_size, rnn_hidden_dim)

        # Final prediction using fully connected layer
        output = self.fc(last_rnn_out)  # (batch_size, output_dim)
        output = torch.cat((output, game_metadata), dim=1)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


def test_loss(model: nn.Module, test_loader: DataLoader, loss_function: nn.modules.loss._Loss) -> float:
    """Get the current loss of the data in the test_loader"""

    # Set model to evaluation mode
    model.eval()
    test_loss = 0

    # No backpropagation calculations needed
    with torch.no_grad():
        for data, moves, board_states, target, lengths in test_loader:

            # Move data to GPU if applicable
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            lengths = lengths.to(device).float()
            target = target.to(device).long()

            model.eval()

            # Predict the data, get the loss based on the prediction
            output = model(data, moves, board_states, lengths)
            loss = loss_function(output, target.long())
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
        for data, moves, board_states, target, lengths in test_loader:

            # Move data to GPU if applicable
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            lengths = lengths.to(device).float()  # Use lengths directly

            model.eval()
            # Predict the data
            output = model(data, moves, board_states, lengths)

            # Index of the highest value in the predicted tensor will be our chosen class
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
        # print(f"Starting epoch {e}")

        # Set model to training mode
        model.train()

        # Cumulative loss across all batches for the current epoch (display purposes only)
        current_epoch_train_loss = 0

        for i, (data, moves, board_states, target, lengths) in enumerate(train_loader):
            # Move data to GPU if applicable
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            target = target.to(device).long()
            lengths = lengths.to(device).float()

            # Zero the gradients to prevent interference from previous iterations
            optimizer.zero_grad()

            # Predict the data
            output = model(data, moves, board_states, lengths)

            # print(f"Output shape: {output.shape}")
            # print(f"Output sample: {output[0]}")

            # Calculate the loss
            loss = loss_function(output, target.long())

            # Backpropagate to update parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_epoch_train_loss += loss.item()

        # Save average train and test loss for later graphing
        train_losses.append(current_epoch_train_loss / len(train_loader))
        test_losses.append(test_loss(model, test_loader, loss_function=loss_function))

        # For debug
        if e % print_every == 0:
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


def collate_fn(batch):
    # Assuming batch is a list of tuples: (data, moves, board_states, labels)

    # Extract individual components from the batch
    data = [item[0] for item in batch]
    moves = [item[1] for item in batch]
    board_states = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    # Check for zero values in board_states and replace them with a suitable value (e.g., 1)
    # This assumes the board state should not have 0 values and replacing 0 with 1 makes sense in your case
    # Check for zero values in board_states and replace them with a suitable value (e.g., 1)
    for i, bs in enumerate(board_states):
        if len(bs) == 0:
            print(f"Warning: board_state at index {i} is empty.")
        # Check if the elements in the board state are valid (no None or invalid data)
        if any(val is None for row in bs for val in row):
            print(f"Warning: board_state at index {i} contains None values.")

    lengths = torch.tensor([len(bs) for bs in board_states], dtype=torch.int32)

    # Pad sequences to the same length (you could also pad moves if necessary)
    data_padded = pad_sequence([torch.tensor(d) for d in data], batch_first=True, padding_value=0)
    moves_padded = pad_sequence([torch.tensor(m) for m in moves], batch_first=True, padding_value=0)
    board_states_padded = pad_sequence([bs.clone().detach() for bs in board_states], batch_first=True, padding_value=0)

    # Stack labels into a single tensor
    labels_stacked = torch.stack([l.clone().detach() for l in labels])

    return data_padded, moves_padded, board_states_padded, labels_stacked, lengths


def printPerformaceMetrics(model, test_loader):
    y_pred = predict(model, test_loader)  # Get predicted labels
    y_test = []

    # Extract ground truth labels and flatten them
    for data, moves, board_states, target, lengths in test_loader:
        y_test.extend(target.tolist())  # Convert target tensor to a list of labels

    # Now you can use these for metrics
    pm.print_metrics(y_pred, y_test)
