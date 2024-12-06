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
elif torch.backends.mps.is_available():
    print("MPS available")
    device = torch.device("mps")
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
        self.labels = torch.tensor(game_data["Result"].to_numpy(), dtype=torch.float32)

        # Normalize moves data
        def normalize_moves(moves_df):
            # Scale Eval using StandardScaler
            moves_df = moves_df.fillna(0)
            scaler = StandardScaler()
            moves_df["Eval"] = scaler.fit_transform(moves_df["Eval"].values.reshape(-1, 1)).flatten()
            
            # Normalize Time by dividing by initial time
            initial_time = moves_df["Time"].iloc[0]
            moves_df["Time"] = moves_df["Time"] / initial_time
            
            # Convert board states
            moves_df["Board State"] = moves_df.apply(lambda row: fh.board_fen_to_image(row["Board State"]), axis=1)
            return moves_df

        game_data["Moves"] = game_data.apply(lambda row: normalize_moves(row["Moves"]), axis=1)
        
        self.board_states = game_data.apply(lambda row: row["Moves"]["Board State"].to_list(), axis=1)
        self.moves = game_data.apply(lambda row: row["Moves"].drop(columns=["Board State"]).to_numpy(), axis=1).to_numpy()

        self.game_metadata = game_data.drop(columns=["Moves", "Result"]).to_numpy()

        # Scale game metadata
        self.game_metadata = StandardScaler().fit_transform(self.game_metadata)

        self.move_limit = only_use_first_X_moves

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert to float32 when creating tensor
        board_state_tensor = torch.tensor(np.array(self.board_states[idx]), dtype=torch.float32)
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

        metadata_per_move = len(fh.MOVE_HEADER_NAMES) - 1
        metadata_per_game = len(fh.HEADERS_TO_KEEP) - 1

        # Simplified CNN
        self.board_cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        # Reduced CNN output dimension
        self.flatten_cnn_output = nn.Linear(128 * 2 * 2, 256)

        # Simplified LSTM
        self.rnn = nn.LSTM(
            input_size=256 + metadata_per_move,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Simplified fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),  # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256 + metadata_per_game, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fc3 = nn.Linear(128, 3)

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
    learning_rate: float = 0.0003,  # Reduced initial learning rate
    print_every: int = 100,
) -> tuple[list[float], list[float]]:
    train_losses: list[float] = []
    test_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Increased weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.7,    # More gradual reduction
        patience=10,    # Wait longer before reducing
        verbose=True,
        min_lr=1e-6    # Don't let LR get too small
    )

    for e in range(epoch):
        # Training loop remains the same...
        model.train()
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

        # Calculate average losses for this epoch
        avg_train_loss = current_epoch_train_loss / len(train_loader)
        avg_test_loss = test_loss(model, test_loader, loss_function=loss_function)
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Step the scheduler based on validation loss
        scheduler.step(avg_test_loss)

        # For debug
        if e % print_every == 0:
            print(f"Epoch {e} training loss: {avg_train_loss}")
            print(f"Epoch {e} testing loss: {avg_test_loss}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

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

    # Pad sequences and ensure float32 dtype
    data_padded = pad_sequence([torch.tensor(d, dtype=torch.float32) for d in data], batch_first=True, padding_value=0)
    moves_padded = pad_sequence([torch.tensor(m, dtype=torch.float32) for m in moves], batch_first=True, padding_value=0)
    board_states_padded = pad_sequence([bs.clone().detach().to(torch.float32) for bs in board_states], batch_first=True, padding_value=0)

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
