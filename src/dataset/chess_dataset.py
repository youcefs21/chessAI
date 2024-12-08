import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset
import torch
import numpy as np

from src.dataset.helpers import board_fen_to_image


def arr_to_imgs(arr):
    return torch.tensor(np.array([board_fen_to_image(x) for x in arr]), dtype=torch.float)


def afloat(arr):
    return torch.tensor(np.array([float(x) for x in arr]), dtype=torch.float32)


class ChessDataset(Dataset):
    def __init__(self, df: pd.DataFrame, move_limit: int | None = None):
        self.df = df
        self.move_limit = move_limit
        self.boards = df["Board"].apply(arr_to_imgs).to_numpy()
        self.times = df["Time"].apply(afloat).to_numpy()
        self.player = df["Player"].apply(afloat).to_numpy()
        self.labels = torch.tensor(df["Result"].to_numpy(), dtype=torch.float32)
        self.game_metadata = df.drop(columns=["Board", "Eval", "Raw Eval", "Time", "Player"]).to_numpy()
        self.game_metadata = StandardScaler().fit_transform(self.game_metadata)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        times = self.times[idx]
        boards = self.boards[idx]
        player = self.player[idx]
        if self.move_limit is not None:
            times = times[: self.move_limit]
            boards = boards[: self.move_limit]
            player = player[: self.move_limit]

        return self.game_metadata[idx], times, boards, player, self.labels[idx]
