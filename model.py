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


class ChessDataset(Dataset):
    def __init__(self, observations: list, labels: list[int]):
        self.observations = observations
        self.labels = labels

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]

class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Define layers

    def forward(self, x):
        # Define a view to shape the data input
        # Define the forward pass through layers