import os

from chess.pgn import Game
from huggingface_hub import HfApi
import logging
import pandas as pd
from chess import pgn, Board
import zstandard as zstd
import io

logger = logging.getLogger("chessAI")


class ChessDataset:
    def __init__(self, repo_id="Youcef/chessGames"):
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
            self.df_train = pd.read_parquet(f"hf://{repo_id}/train.parquet")
        else:
            logger.info("Dataset does not exist, creating...")
            self.create_dataset()

    def create_dataset(self, path="data/lichess_db_standard_rated_2024-11.pgn.zst"):
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
                        logger.info(
                            f"Processed {total_games} games. Valid: {len(games)}, Invalid: {invalid_count}, Ratio: {len(games)/(invalid_count+1):.2f}"
                        )

                    if self.is_valid_game(game):
                        games.append(game)
                    else:
                        invalid_count += 1

    def is_valid_game(self, game: Game) -> bool:
        first_move = game.next()
        time_parts = game.time_control().parts
        if len(time_parts) != 1:
            return False
        time = time_parts[0]

        return (
            # needs to be a nice normal game
            game.headers.get("Termination") == "Normal"
            and first_move is not None
            # standard time rules 60+0 time rules
            and time.time == 60
            and time.increment == 0
            and time.delay == 0
            # make sure the clock for each turn is present
            and "clk" in first_move.comment
        )
