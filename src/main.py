import torch

from dataset.chess_dataframe import ChessDataFrame, Sizes
from dataset.chess_dataset import ChessDataset
from logging_config import setup_logging
from models.rnn_cnn import ChessNN
from torch.utils.data import DataLoader, random_split

from models.rnn_cnn_helpers import collate_fn
from models.simple_rnn import train_rnn

# consts
batch_size = 32
size = Sizes.mid
train_size = size.value * 0.8
validation_size = int(0.2 * train_size)
train_size_split = train_size - validation_size

# Setup logging
logger = setup_logging()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize the chess dataframe
chess_df = ChessDataFrame(size=size)
logger.info("Successfully initialized ChessDataFrame")

rnn = train_rnn(chess_df.df_train, 20)

# Initialize datasets
# train_dataset = ChessDataset(chess_df.df_train, move_limit=10)
# test_dataset = ChessDataset(chess_df.df_test, move_limit=10)
# logger.info("Successfully initialized ChessDataset")


# train_set_split, validation_set_split = random_split(train_dataset, [train_size_split, validation_size], generator=torch.Generator().manual_seed(42))
#
#
# # Initialize the model
# model = ChessNN()
# model.to(device)
#
#
# # init the data loaders
# train_loader = DataLoader(train_set_split, batch_size=batch_size, collate_fn=collate_fn)
# validation_loader = DataLoader(validation_set_split, batch_size=batch_size, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
