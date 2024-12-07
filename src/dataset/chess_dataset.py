import os
from huggingface_hub import HfApi
import logging
import pandas as pd
from chess import pgn
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
    def __init__(self, repo_id="Youcef/chessGames", size=Sizes.smol):
        self.size = size

        logger.info("Initializing ChessDataset")
        api = HfApi()

        # create dataset repo if it doesn't already exist
        if not api.repo_exists(repo_id):
            logger.info(f"Repository {repo_id} does not exist, creating...")
            # api.create_repo(repo_id=repo_id, repo_type="dataset")
            # logger.info(f"Successfully created repository {repo_id}")
        else:
            logger.info(f"Repository {repo_id} already exists")

        # try reading dataset
        if api.file_exists(repo_id, "train.parquet"):
            logger.info("Dataset already exists, loading...")
            self.df_train = pd.read_parquet(f"hf://{repo_id}/train_{size.name}.parquet")
            self.df_test = pd.read_parquet(f"hf://{repo_id}/test_{size.name}.parquet")
        else:
            logger.info("Dataset does not exist, creating...")
            self.create_dataset()

    def create_dataset(
        self,
        path="data/lichess_db_standard_rated_2024-11.pgn.zst",
    ):
        if not os.path.exists(path):
            logger.error(f"File {path} does not exist")
            raise FileNotFoundError(f"File {path} does not exist")

        games = []
        invalid_count = 0
        total_games = 0
        with open(path, "rb") as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                # Wrap the binary stream with TextIOWrapper to get a text stream
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                while True:
                    game = pgn.read_game(text_stream)
                    if game is None:
                        break

                    total_games += 1
                    if total_games % 1000 == 0:
                        logger.info(f"Read {total_games} games. Valid: {len(games)}, Invalid: {invalid_count}, Ratio: {len(games)/(invalid_count+1):.2f}")

                    if is_valid_game(game):
                        games.append(game)
                    else:
                        invalid_count += 1

                    if len(games) >= self.size.value:
                        break

        games_pd = []
        logger.info(f"Preprocessing {len(games)} games...")
        for i, game in enumerate(games):
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
