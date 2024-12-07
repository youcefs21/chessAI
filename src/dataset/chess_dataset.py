import os
from typing import Optional, TextIO

from huggingface_hub import HfApi
import logging
import pandas as pd
from chess import pgn
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import zstandard as zstd
import io
from enum import Enum

from src.dataset.helpers import is_valid_game, preprocess_game, HEADERS_TO_KEEP


class Sizes(Enum):
    """
    Fixed size options for the chess dataset.
    These values should remain constant - add new enum values if different sizes are needed.
    Used for naming dataset files.
    """

    extra_smol = 1000
    smol = 10_000
    mid = 100_000
    large = 1_000_000


logger = logging.getLogger("chessAI")


class ChessDataset:
    df_train: DataFrame
    df_test: DataFrame

    def __init__(self, repo_id="Youcef/chessGames", size=Sizes.smol):

        self.size = size
        self.repo_id = repo_id
        self._total_games = 0
        self._invalid_count = 0
        self._games = []

        logger.info("Initializing ChessDataset")
        api = HfApi()

        # create dataset repo if it doesn't already exist
        if not api.repo_exists(repo_id):
            logger.info(f"Repository {repo_id} does not exist, creating...")
            api.create_repo(repo_id=repo_id, repo_type="dataset")
            logger.info(f"Successfully created repository {repo_id}")
        else:
            logger.info(f"Repository {repo_id} already exists")

        # try reading dataset
        if api.file_exists(repo_id, f"train_{size.name}.parquet"):
            logger.info("Dataset already exists, loading...")
            self.load_dataset()
        else:
            logger.info("Dataset does not exist, creating...")
            self.create_dataset()
            self.save_dataset()

    def save_dataset(self):
        self.df_train.to_parquet(f"hf://{self.repo_id}/train_{self.size.name}.parquet")
        self.df_test.to_parquet(f"hf://{self.repo_id}/test_{self.size.name}.parquet")

    def load_dataset(self):
        self.df_train = pd.read_parquet(f"hf://{self.repo_id}/train_{self.size.name}.parquet")
        self.df_test = pd.read_parquet(f"hf://{self.repo_id}/test_{self.size.name}.parquet")

    def process_stream(self, text_stream: TextIO):
        while True:
            game = pgn.read_game(text_stream)
            if game is None:
                break

            self._total_games += 1
            if self._total_games % 1000 == 0:
                logger.info(f"Read {self._total_games} games. Valid: {len(self._games)}, Invalid: {self._invalid_count}, Ratio: {len(self._games) / (self._invalid_count + 1):.2f}")

            if is_valid_game(game):
                self._games.append(game)
            else:
                self._invalid_count += 1

            if len(self._games) >= self.size.value:
                return True

        return False

    def create_dataset(
        self,
        read_compressed=False,
    ):

        done = False
        if read_compressed:
            # find compressed file in data/
            compressed_file = next((file for file in os.listdir("data/") if file.endswith(".zst")), None)
            if compressed_file is None:
                logger.error("No compressed file found in data/")
                raise FileNotFoundError("No compressed file found in data/")

            with open(f"data/{compressed_file}", "rb") as compressed_file:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(compressed_file) as reader:
                    # Wrap the binary stream with TextIOWrapper to get a text stream
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    done = self.process_stream(text_stream)
        else:
            # read all .pgn files in data/ until done
            done = False
            for file in os.listdir("data/"):
                if file.endswith(".pgn"):
                    logger.info(f"Reading {file}...")
                    with open(f"data/{file}", "r") as pgn_file:
                        done = self.process_stream(pgn_file)

                if done:
                    break

        games_pd = []
        logger.info(f"Preprocessing {len(self._games)} games...")
        for i, game in enumerate(self._games):
            games_pd.append(preprocess_game(game))
            if (i + 1) % 500 == 0:
                logger.info(f"Preprocessed {i + 1} games")

        game_data = pd.DataFrame(games_pd, columns=HEADERS_TO_KEEP + ["Moves"])

        # split into train and test
        split_size = 0.2
        logger.info(f"Splitting {len(game_data)} games into train and test (test size: {split_size})...")
        train_data, test_data = train_test_split(game_data, test_size=split_size, random_state=42)

        self.df_train = train_data
        self.df_test = test_data
