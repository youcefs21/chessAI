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


class MoveDataset(Dataset):
    def __init__(self, moves: pd.DataFrame):
        self.moves = moves
        self.moves["Board State"] = self.moves.apply(lambda row: fh.board_fen_to_image(row["Board State"]), axis=1)

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> pd.Series:
        return self.moves.iloc[idx]


class ChessDataset(Dataset):
    def __init__(self, game_data: pd.DataFrame):
        self.labels: np.ndarray = game_data["Result"].values
        self.observations = game_data.drop(columns=["Result"])
        # print(self.observations["Moves"])
        self.observations["Moves"] = self.observations.apply(lambda row: MoveDataset(row["Moves"]), axis=1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[pd.Series, int]:
        return self.observations.iloc[idx], self.labels[idx]


# TESTing dataset
game_data = fh.pgn_file_to_dataframe("Data/2024-08/xaa.pgn")
# print(game_data)
print(game_data.iloc[0]["Moves"]["Board State"])
dataset = ChessDataset(game_data)
print(dataset[0][0]["Moves"][0]["Board State"])


class MovesRNN(nn.Module):
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

        pass
        # Define a view to shape the data input
        # Define the forward pass through layers
