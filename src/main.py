import torch

from dataset.chess_dataframe import ChessDataFrame, Sizes
from logging_config import setup_logging

from models.simple_rnn import train_rnn, evaluate_model

# consts
size = Sizes.mid_no_time_restriction

# Setup logging
logger = setup_logging()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize the chess dataframe
chess_df = ChessDataFrame(size=size)
logger.info("Successfully initialized ChessDataFrame")

rnn = train_rnn(chess_df.df_train, 20)

metrics = evaluate_model(rnn, chess_df.df_test, "Chess RNN Evaluation")
logger.info(metrics)
