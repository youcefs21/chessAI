import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import numpy as np

from dataset import ChessDataset
from logging_config import setup_logging

# Setup logging
logger = setup_logging()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# I need to make csv batches of data and uploads it to huggingface
dataset = ChessDataset()
logger.info("Successfully initialized ChessDataset")