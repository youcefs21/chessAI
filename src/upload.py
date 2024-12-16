import torch

from dataset.chess_dataframe import ChessDataFrame, Sizes
from logging_config import setup_logging


logger = setup_logging()

df = ChessDataFrame(size=Sizes.large_no_time_or_ending_restriction)
df.create_dataset()
df.save_dataset()
