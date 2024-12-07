import torch

from dataset.chess_dataframe import ChessDataFrame
from logging_config import setup_logging

# Setup logging
logger = setup_logging()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# I need to make csv batches of data and uploads it to huggingface
chess_df = ChessDataFrame()
logger.info("Successfully initialized ChessDataset")
