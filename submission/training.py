import os
import chess
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import torch

import logging
import sys
from colorlog import ColoredFormatter
from enum import Enum
from huggingface_hub import HfApi
from chess import pgn
from chess.pgn import Game
from typing import Optional, TextIO
import zstandard as zstd
from sklearn.model_selection import train_test_split
import io
from keras.layers import Dropout
from keras.regularizers import l2
from keras.layers import SimpleRNN, Dense
import keras


def setup_logging():
    """
    Setup logger for pretty timestamped logs.
    """
    # create color formatter for console
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s %(bold_white)s%(message)s%(reset)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    
    # configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # create logger
    logger = logging.getLogger('chessAI')
    logger.setLevel(logging.INFO)
    
    # remove any existing handlers
    logger.handlers = []
    
    # add console handler
    logger.addHandler(console_handler)
    
    return logger 

# Setup logging
logger = setup_logging()


class Sizes(Enum):
    """
    Fixed size options for the chess dataset.
    These values should remain constant - add new enum values if different sizes are needed.
    Used for naming dataset files on Hugging Face.
    """

    smol = 10_000
    mid = 100_000
    large = 1_000_000

HEADERS = [
    # Game headers
    "Result",
    "WhiteElo",
    "BlackElo",
    # Move headers
    "Player",
    "Time",
    "Eval",
    "Raw Eval",
    "Board",
    "UCI",  # e.g., 'e2e4'
    "MovingPiece",  # e.g., 'P' for white pawn, 'p' for black pawn
    "CapturedPiece",  # e.g., 'n' for black knight, None if no capture
]

PIECE_CHANNELS = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}

def board_fen_to_image(board_fen: str):
    """
    Preprocess a chess board to a 12-channel image (one channel per piece).
    Input is the board state expressed as a FEN string.

    There are 12 planes: Each plane corresponds to one type of piece (e.g., white pawn, black rook).
    """

    pieces = board_fen.split(" ")[0]
    rows = pieces.split("/")
    board = np.zeros((12, 8, 8), dtype=np.int8)

    # Populate piece planes
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
                continue

            piece_index = PIECE_CHANNELS[char]
            board[piece_index - 1, i, j] = 1
            j += 1

    return board


def move_to_tuple(
    move: pgn.ChildNode,
    time_control: int,
) -> tuple[float, float | None, str | None, str, str, str, str | None]:
    """
    Converts move data to a tuple of the following:
    - Clock time left for the player (minutes)
    - Evaluation of the move (if available)
    - Raw evaluation string
    - Board FEN
    - Move UCI string (e.g., 'e2e4' for pawn e2 to e4)
    - Moving piece type (e.g., 'P' for pawn, 'N' for knight)
    - Captured piece type (None if no capture)
    """
    # For time conversions to seconds
    ftr = [3600, 60, 1]

    # Dissect the comment part of the move to extract eval and clk
    split_comment = move.comment.split("] [")

    # If no eval or clk data, will show up as None/NaN
    eval_c: Optional[float] = None
    clk_c: Optional[int] = None
    eval_c_s: Optional[str] = None
    for c in split_comment:
        if "eval" in c:
            eval_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            if "#" not in eval_c_s:
                eval_c = float(eval_c_s)
                continue

            # If a side has mate in x moves, they automatically get a min. rating of 40
            # Otherwise, fewer moves till mate => higher advantage for that side
            # (Somewhat arbitrarily chosen number)
            mate_in = eval_c_s.split("#")[1].replace("-", "")
            eval_c = max(320 - float(mate_in) * 10, 40)

            if "-" in eval_c_s:
                eval_c = -eval_c
            continue

        if "clk" in c:
            clk_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            # Convert string formatted time to seconds
            clk_c = sum([a * b for a, b in zip(ftr, map(int, clk_c_s.split(":")))])

    # Get move details
    parent_board = move.parent.board()  # Get board BEFORE the move
    board = move.board()
    uci = move.move.uci()
    from_square = uci[:2]
    to_square = uci[2:4]

    # Get piece being moved
    piece = parent_board.piece_at(chess.parse_square(from_square))
    if piece is None:
        raise ValueError(f"No piece found at square {from_square} in position {board.fen()}")
    moving_piece = piece.symbol()

    # Get captured piece (if any)
    captured_piece = None
    if parent_board.is_capture(move.move):
        captured_square = to_square
        if parent_board.is_en_passant(move.move):
            # Adjust captured square for en passant
            captured_square = to_square[0] + ("5" if moving_piece.isupper() else "4")
        captured_piece_obj = parent_board.piece_at(chess.parse_square(captured_square))
        if captured_piece_obj is not None:
            captured_piece = captured_piece_obj.symbol()

    return clk_c / time_control, eval_c, eval_c_s, board.fen(), uci, moving_piece, captured_piece


def preprocess_game(game: Game):
    # Get game headers
    result = 1 if game.headers.get("Result") == "1-0" else 0
    white_elo = int(game.headers.get("WhiteElo"))
    black_elo = int(game.headers.get("BlackElo"))
    time_control = int(game.headers.get("TimeControl").split("+")[0])

    # Initialize game data dictionary
    game_data = {
        "Result": result,
        "WhiteElo": white_elo,
        "BlackElo": black_elo,
        "Player": [],
        "Time": [],
        "Eval": [],
        "Raw Eval": [],
        "Board": [],
        "UCI": [],
        "MovingPiece": [],
        "CapturedPiece": [],
    }

    first_move_player = 0
    for move in game.mainline():
        time, eval_c, raw_eval, board_state, uci, moving_piece, captured_piece = move_to_tuple(move, time_control)
        if eval_c is None:
            break

        # Append each value to its respective list in the dict
        game_data["Player"].append(first_move_player)
        game_data["Time"].append(time)
        game_data["Eval"].append(eval_c)
        game_data["Raw Eval"].append(raw_eval)
        game_data["Board"].append(board_state)
        game_data["UCI"].append(uci)
        game_data["MovingPiece"].append(moving_piece)
        game_data["CapturedPiece"].append(captured_piece)

        first_move_player = (first_move_player + 1) % 2

    # Convert dict to list using HEADERS order
    return [game_data[header] for header in HEADERS]

ELO_SCALER = MinMaxScaler()

def encode_piece(pieces):
    """
    One-hot encode pieces. If piece is None, encode as [0,0,0,0,0,0].
    Returns a 2D array where each row is a one-hot vector.
    """
    num_pieces = len(PIECE_CHANNELS)
    encoded = np.zeros((len(pieces), num_pieces), dtype=np.int8)

    for i, piece in enumerate(pieces):
        if piece is not None:  # Only encode if piece exists
            channel = PIECE_CHANNELS[piece.upper()]
            encoded[i, channel] = 1

    return encoded


def aint(arr):
    return np.array([[int(x)] for x in arr], dtype=np.int8)


def encode_uci(uci):
    """
    Convert UCI move notation (e.g. 'e2e4') into a 128-length one-hot array
    representing the from and to squares (64 bits each)
    """
    # Chess square mapping: a1=0, b1=1, ..., h8=63
    file_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    # Calculate from_square index (0-63)
    from_file = file_map[uci[0]]
    from_rank = int(uci[1]) - 1
    from_square = from_rank * 8 + from_file

    # Calculate to_square index (0-63)
    to_file = file_map[uci[2]]
    to_rank = int(uci[3]) - 1
    to_square = to_rank * 8 + to_file

    # Create one-hot encoding
    encoded = np.zeros(128, dtype=np.int8)
    encoded[from_square] = 1  # First 64 bits for from_square
    encoded[to_square + 64] = 1  # Last 64 bits for to_square

    return encoded


def encode_ucis(ucis):
    return np.array([encode_uci(uci) for uci in ucis])


def log_scale(x, global_min, global_max):
    # Shift all values to be positive
    shifted = x - global_min + 1  # Add 1 to avoid log(0)
    shifted_max = global_max - global_min + 1

    # Apply logarithmic scaling
    log_scaled = np.log(shifted) / np.log(shifted_max)

    # For negative evaluations (original values < 0), make the log scaling negative
    log_scaled = np.where(x < 0, -log_scaled, log_scaled)

    return log_scaled

def preprocess_data(data):
    """
    Preprocess the data for the RNN model.
    """
    # Normalize Elo ratings
    elo_data = data[["WhiteElo", "BlackElo"]].values
    normalized_elos = ELO_SCALER.fit_transform(elo_data)

    # Existing preprocessing
    players = data["Player"].apply(aint).to_numpy()
    moving_pieces = data["MovingPiece"].apply(encode_piece).to_numpy()
    captured_pieces = data["CapturedPiece"].apply(encode_piece).to_numpy()
    uci_moves = data["UCI"].apply(encode_ucis).to_numpy()
    results = data["Result"].to_numpy()
    evals = data["Eval"].to_numpy()

    # Create a list to store all games
    games = []

    # can we scale all the evals at once?
    # find the global min and max of the evals
    # Handle each game's evals separately since they have different lengths
    min_evals = []
    max_evals = []
    for eval_array in evals:
        if len(eval_array) > 0:  # Only process non-empty arrays
            min_evals.append(np.min(eval_array))
            max_evals.append(np.max(eval_array))

    global_min = np.min(min_evals) if min_evals else 0
    global_max = np.max(max_evals) if max_evals else 0
    logger.debug("global_min: %s", global_min)
    logger.debug("global_max: %s", global_max)

    # Iterate through each game's data
    for i in range(len(players)):
        # Create array of normalized Elos for this game
        game_elos = np.tile(normalized_elos[i], (len(players[i]), 1))

        game_evals = evals[i]
        game_evals = log_scale(game_evals, global_min, global_max)
        game_evals = game_evals.reshape(-1, 1)

        # Concatenate all features including Elos and eval scores
        game_features = np.concatenate(
            [
                players[i],  # Player info
                moving_pieces[i],  # Moving piece info
                captured_pieces[i],  # Captured piece info
                uci_moves[i],  # UCI move encoding
                game_evals,  # Eval scores for each move
                game_elos,  # Normalized Elo ratings (2 values per move)
            ],
            axis=1,
        )

        games.append(game_features)

    return games, results

def is_valid_game(game: Game) -> bool:
    game_length = sum(1 for _ in game.mainline())
    first_move = game.next()
    time_parts = game.time_control().parts
    if len(time_parts) != 1:
        return False
    time = time_parts[0]

    return (
        # needs to be a nice normal game with 20-60 moves
        # game.headers.get("Termination") == "Normal"
        # and game.headers.get("Result") != "1/2-1/2"  # exclude draws
        game.headers.get("Result") != "1/2-1/2"  # exclude draws
        and first_move is not None
        # and 20 <= game_length <= 60
        and 10 <= game_length <= 60
        # standard time rules 60+0 time rules
        # and time.time == 60
        and time.increment == 0
        and time.delay == 0
        # make sure the clock for each turn is present
        and "clk" in first_move.comment
        and "eval" in first_move.comment
    )


class ChessDataFrame:
    """
    A class that filters and partially preprocesses the chess dataset,
    and caches the result on Hugging Face.

    If the dataset file is already cached on Hugging Face, 
    it will be loaded instead of processed again.
    """
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
        if not api.repo_exists(repo_id, repo_type="dataset"):
            logger.info(f"Repository {repo_id} does not exist, creating...")
            api.create_repo(repo_id=repo_id, repo_type="dataset")
            logger.info(f"Successfully created repository {repo_id}")
        else:
            logger.info(f"Repository {repo_id} already exists")

        # try reading dataset
        if api.file_exists(repo_id, f"train_{size.name}.parquet", repo_type="dataset"):
            logger.info("Dataset already exists, loading...")
            self.load_dataset()
        else:
            logger.info("Dataset does not exist, creating...")
            self.create_dataset()
            self.save_dataset()

    def load_dataset(self):
        self.df_train = pd.read_parquet(f"hf://datasets/{self.repo_id}/train_{self.size.name}.parquet")
        logger.info(f"Loaded {len(self.df_train)} train games")
        self.df_test = pd.read_parquet(f"hf://datasets/{self.repo_id}/test_{self.size.name}.parquet")
        logger.info(f"Loaded {len(self.df_test)} test games")

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

        game_data = pd.DataFrame(games_pd, columns=HEADERS)

        # split into train and test
        split_size = 0.2
        logger.info(f"Splitting {len(game_data)} games into train and test (test size: {split_size})...")
        train_data, test_data = train_test_split(game_data, test_size=split_size, random_state=42)

        self.df_train = train_data
        self.df_test = test_data

class ChessRNN:
    def __init__(self, sequence_length=10, LR=0.001, validation_split=0.2, epochs=10, batch_size=32):
        """
        Initialize the RNN model
        sequence_length: number of moves to consider from start of game
        """
        logger.info(f"Initializing ChessRNN with sequence length {sequence_length}")
        self.sequence_length = sequence_length
        self.model = None
        self.LR = LR
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size

    def build_model(self, input_shape):
        """
        Build the RNN model architecture with improved stability
        """
        logger.info(f"Building model with input shape {input_shape}")

        # create model with Input layer explicitly
        inputs = keras.Input(shape=input_shape)
        metadata_inputs = inputs[:, :, -2:]  # elo ratings are the last 2 features
        metadata_inputs = metadata_inputs[:, 0, :]

        x = Dropout(0.1)(inputs[:, :, :-2])

        # First RNN layer
        # x = keras.layers.LSTM(
        x = SimpleRNN(
            128,
            return_sequences=True,
            kernel_regularizer=l2(self.LR),
            recurrent_regularizer=l2(self.LR),
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            activation="tanh",
        )(x)
        x = Dropout(0.2)(x)

        # second RNN layer
        x = SimpleRNN(
            64,
            kernel_regularizer=l2(self.LR),
            recurrent_regularizer=l2(self.LR),
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            activation="tanh",
        )(x)
        x = Dropout(0.2)(x)

        # dense layers
        x = keras.layers.concatenate([x, metadata_inputs])  # combine RNN output with Elo ratings
        x = Dense(32, activation="relu", kernel_regularizer=l2(self.LR))(x)
        x = Dropout(0.1)(x)
        x = Dense(16, activation="relu", kernel_regularizer=l2(self.LR))(x)
        outputs = Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # configure Adam optimizer with default hyperparameters and gradient clipping
        optimizer = keras.optimizers.Adam(
            learning_rate=self.LR,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0,
        )

        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        logger.info("Model compiled successfully")

    def prepare_sequences(self, games, results):
        """
        Prepare move sequences for training/prediction
        Pad or truncate games to sequence_length
        """
        logger.info("Preparing sequences...")
        X = []
        y = []

        for game, result in zip(games, results):
            if len(game) >= self.sequence_length:
                # Take first sequence_length moves
                sequence = game[: self.sequence_length]
                # print("result: ", result)
                # print(decode_sequence_element(sequence[0]))
                X.append(sequence)
                y.append(result)

        logger.info(f"Prepared {len(X)} sequences")
        return np.array(X), np.array(y)

    def train(self, games, results, model_name):
        """
        Train the model with improved training process
        """
        logger.info("Starting model training...")
        X, y = self.prepare_sequences(games, results)

        if self.model is None:
            input_shape = (self.sequence_length, X.shape[2])
            self.build_model(input_shape)

        callbacks = [
            # early stopping if val_loss doesn't improve for 15 epochs
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
            ),
            # reduce learning rate when plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1,
            ),
            # model checkpoint with .keras extension
            keras.callbacks.ModelCheckpoint(
                model_name, 
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # class weights to handle imbalanced data
        class_counts = np.bincount(y.astype(int))
        total = len(y)
        class_weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1]),
        }

        logger.info(f"Training model with {len(X)} samples")
        history = self.model.fit(
            X,
            y,
            validation_split=self.validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            shuffle=True,
        )

        logger.info("Model training completed")
        return history

    def predict(self, games):
        """
        Predict outcomes for new games
        """
        logger.info(f"Making predictions for {len(games)} games")
        X, _ = self.prepare_sequences(games, np.zeros(len(games)))
        predictions = self.model.predict(X)
        logger.info("Predictions completed")
        return predictions

    def evaluate(self, games, results):
        """
        Evaluate model performance on test data
        """
        logger.info("Evaluating model performance...")
        X, y = self.prepare_sequences(games, results)
        metrics = self.model.evaluate(X, y)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy
    """
    import matplotlib.pyplot as plt

    # Loss Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

def train_rnn(data, steps_per_game, model_name="best_model.keras", plot=True, LR = 0.001, epochs=10, batch_size=32):
    logger.info(f"Training RNN with {steps_per_game} steps per game")

    rnn = ChessRNN(steps_per_game, LR, 0.2, epochs, batch_size)
    games, results = preprocess_data(data)

    logger.info(f"Preprocessed {len(games)} games")
    history = rnn.train(games, results, model_name=model_name)

    if plot:
        plot_training_history(history)
    logger.info("RNN training completed")
    return rnn


if __name__ == "__main__":
    # consts
    size = Sizes.smol

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize the chess dataframe
    chess_df = ChessDataFrame(size=size)
    logger.info("Successfully initialized ChessDataFrame")

    rnn = train_rnn(chess_df.df_train, 30)


#######################################################
### this was used to measure the usefulness of CNNs ###
#######################################################
# class ChessEvalNet(nn.Module):
#     def __init__(self):
#         super(ChessEvalNet, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
#         # Batch normalization layers
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm2d(256)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(256 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         # Convolutional blocks
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
        
#         # Flatten and fully connected layers
#         x = x.view(-1, 256 * 8 * 8)
#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.dropout(self.relu(self.fc2(x)))
#         x = self.fc3(x)
        
#         return x

# # Create model and move to device
# model = ChessEvalNet().to(device)

# # Convert data to tensors and create DataLoader
# X_train = torch.FloatTensor(train_X).to(device)
# y_train = torch.FloatTensor(train_Y).to(device)
# X_test = torch.FloatTensor(test_X).to(device)
# y_test = torch.FloatTensor(test_Y).to(device)

# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# # Loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# def evaluate_model(model_to_eval, data_loader_eval):
#     model_to_eval.eval()
#     eval_total_loss = 0
#     with torch.no_grad():
#         for batch_x, batch_Y in data_loader_eval:
#             batch_outputs = model_to_eval(batch_x)
#             batch_loss = criterion(batch_outputs.squeeze(), batch_Y)
#             eval_total_loss += batch_loss.item()
#     return eval_total_loss / len(data_loader_eval)

# # Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     # Training phase
#     model.train()
#     total_loss = 0
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs.squeeze(), batch_y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     # Calculate losses
#     train_loss = total_loss / len(train_loader)
#     test_loss = evaluate_model(model, test_loader)
    
#     # Print epoch statistics
#     logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


############################
### OLD MILESTONE 2 CODE ###
############################
# class ChessDataset(Dataset):
#     def __init__(self, game_data: pd.DataFrame, only_use_first_X_moves: int | None = None):
#         """dataset for chess games with normalized features"""
#         game_data = game_data.dropna()
#         self.labels = torch.tensor(game_data["Result"].to_numpy(), dtype=torch.float32)

#         # each game has a list of moves -> this function needs to run on each list of moves
#         def normalize_moves(moves_df):
#             # normalize evaluation scores
#             moves_df = moves_df.fillna(0)
#             scaler = StandardScaler()
#             moves_df["Eval"] = scaler.fit_transform(moves_df["Eval"].values.reshape(-1, 1)).flatten()

#             # normalize remaining time relative to starting time
#             initial_time = moves_df["Time"].iloc[0]
#             moves_df["Time"] = moves_df["Time"] / initial_time

#             # convert board states to tensor representation
#             moves_df["Board State"] = moves_df.apply(lambda row: board_fen_to_image(row["Board State"]), axis=1)
#             return moves_df

#         game_data["Moves"] = game_data.apply(lambda row: normalize_moves(row["Moves"]), axis=1)

#         self.board_states = game_data.apply(lambda row: row["Moves"]["Board State"].to_list(), axis=1)
#         self.moves = game_data.apply(lambda row: row["Moves"].drop(columns=["Board State"]).to_numpy(), axis=1).to_numpy()

#         self.game_metadata = game_data.drop(columns=["Moves", "Result"]).to_numpy()

#         # normalize game metadata features
#         self.game_metadata = StandardScaler().fit_transform(self.game_metadata)

#         self.move_limit = only_use_first_X_moves

#     def __len__(self) -> int:
#         return len(self.labels)

#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         board_state_tensor = torch.tensor(np.array(self.board_states[idx]), dtype=torch.float32)
#         moves = self.moves[idx]

#         # restrict to first X moves if needed
#         if self.move_limit is not None:
#             moves = moves[: self.move_limit]
#             board_state_tensor = board_state_tensor[: self.move_limit]

#         return self.game_metadata[idx], moves, board_state_tensor, self.labels[idx]


# class ChessNN(nn.Module):
#     def __init__(self):
#         """neural network for chess game outcome prediction

#         architecture:
#         1. cnn processes board states
#         2. lstm processes move sequences
#         3. fully connected layers combine features for final prediction"""
#         super().__init__()

#         metadata_per_move = len(MOVE_HEADER_NAMES) - 1
#         metadata_per_game = len(HEADERS_TO_KEEP) - 1

#         # process board states with cnn
#         self.board_cnn = nn.Sequential(
#             nn.Conv2d(12, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Dropout2d(0.2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((2, 2)),
#         )

#         # project cnn features to lstm input size
#         self.flatten_cnn_output = nn.Linear(128 * 2 * 2, 256)

#         # process move sequences
#         self.rnn = nn.LSTM(input_size=256 + metadata_per_move, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

#         # classification layers
#         self.fc = nn.Sequential(nn.Linear(256 * 2, 256), nn.ReLU(), nn.Dropout(0.4))

#         # combine with game metadata
#         self.fc2 = nn.Sequential(nn.Linear(256 + metadata_per_game, 128), nn.ReLU(), nn.Dropout(0.4))

#         # final prediction layer
#         self.fc3 = nn.Linear(128, 3)

#     def forward(self, game_metadata, moves, board_states, lengths):
#         batch_size, seq_len, _, _, _ = board_states.size()

#         # process board states
#         cnn_out = self.board_cnn(board_states.view(-1, 12, 8, 8))
#         cnn_out = cnn_out.view(batch_size * seq_len, -1)
#         cnn_out = self.flatten_cnn_output(cnn_out)
#         cnn_out = cnn_out.view(batch_size, seq_len, -1)

#         # combine cnn features with move data
#         combined_features = torch.cat((cnn_out, moves), dim=2)

#         # process move sequences
#         packed_features = pack_padded_sequence(combined_features, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
#         packed_rnn_out, _ = self.rnn(packed_features)
#         rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)

#         # get final states
#         idx = (lengths - 1).clone().detach().to(device=rnn_out.device).unsqueeze(1).unsqueeze(2).expand(-1, 1, rnn_out.size(2)).long()
#         last_rnn_out = rnn_out.gather(1, idx).squeeze(1)

#         # final classification
#         output = self.fc(last_rnn_out)
#         output = torch.cat((output, game_metadata), dim=1)
#         output = self.fc2(output)
#         output = self.fc3(output)

#         return output

# def test_loss(model: nn.Module, test_loader: DataLoader, loss_function: nn.modules.loss._Loss) -> float:
#     """calculate average loss on test data"""
#     model.eval()
#     test_loss = 0

#     with torch.no_grad():
#         for data, moves, board_states, target, lengths in test_loader:
#             # move data to device
#             data = data.to(device).float()
#             moves = moves.to(device).float()
#             board_states = board_states.to(device).float()
#             lengths = lengths.to(device).float()
#             target = target.to(device).long()

#             # set model to eval mode and calculate loss
#             model.eval()
#             output = model(data, moves, board_states, lengths)
#             loss = loss_function(output, target.long())
#             test_loss += loss.item()

#     return test_loss / len(test_loader)


# def predict(model: nn.Module, test_loader: DataLoader) -> list[int]:
#     """generate predictions for test data"""
#     model.eval()
#     predictions = []

#     with torch.no_grad():
#         for data, moves, board_states, target, lengths in test_loader:
#             # move data to device
#             data = data.to(device).float()
#             moves = moves.to(device).float()
#             board_states = board_states.to(device).float()
#             lengths = lengths.to(device).float()

#             # set model to eval mode and make prediction
#             model.eval()
#             output = model(data, moves, board_states, lengths)
#             _, predicted = torch.max(output.data, 1)
#             predictions += predicted.tolist()

#     return predictions


# def train(
#     model: nn.Module,
#     loss_function: nn.modules.loss._Loss,
#     train_loader: DataLoader,
#     test_loader: DataLoader,
#     epoch: int,
#     learning_rate: float,
#     print_every: int,
# ) -> tuple[list[float], list[float], list[float], list[float]]:
#     """train model and track metrics"""
#     train_losses: list[float] = []
#     test_losses = []
#     train_accuracy = []
#     test_accuracy = []

#     # setup optimizer with weight decay for regularization
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

#     # reduce learning rate when validation loss plateaus
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=0.7,
#         patience=10,
#         verbose=True,
#         min_lr=1e-6,
#     )

#     for e in range(epoch):
#         model.train()
#         current_epoch_train_loss = 0

#         for i, (data, moves, board_states, target, lengths) in enumerate(train_loader):
#             # move data to device
#             data = data.to(device).float()
#             moves = moves.to(device).float()
#             board_states = board_states.to(device).float()
#             target = target.to(device).long()
#             lengths = lengths.to(device).float()

#             optimizer.zero_grad()
#             output = model(data, moves, board_states, lengths)
#             loss = loss_function(output, target.long())
#             loss.backward()

#             # clip gradients to prevent explosion
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             current_epoch_train_loss += loss.item()

#         # calculate metrics
#         avg_train_loss = current_epoch_train_loss / len(train_loader)
#         avg_test_loss = test_loss(model, test_loader, loss_function=loss_function)

#         # save metrics for graphing
#         train_losses.append(avg_train_loss)
#         test_losses.append(avg_test_loss)

#         train_accuracy.append(
#             accuracy(
#                 predict(model, train_loader),
#                 flatten([label[3] for label in train_loader]),
#             )
#         )
#         test_accuracy.append(
#             accuracy(
#                 predict(model, test_loader),
#                 flatten([label[3] for label in test_loader]),
#             )
#         )

#         # reduce learning rate if needed
#         scheduler.step(avg_test_loss)

#         if e % print_every == 0:
#             print(f"epoch {e} training loss: {avg_train_loss}")
#             print(f"epoch {e} testing loss: {avg_test_loss}")
#             print(f"current learning rate: {optimizer.param_groups[0]['lr']}")
#             print(f"train accuracy: {train_accuracy[-1]}")
#             print(f"test accuracy: {test_accuracy[-1]}")

#     # Replace the pickle save with torch.save for the state dict
#     model_save_path = "model.pt"
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved to {model_save_path}")

#     return train_losses, test_losses, train_accuracy, test_accuracy


# def accuracy(y_pred, y_test) -> float:
#     """calculate prediction accuracy"""
#     y_pred = np.array(y_pred)
#     y_test = np.array(y_test)
#     correct = np.sum(y_pred == y_test)
#     total = len(y_test)
#     return correct / total if total > 0 else 0


# def collate_fn(batch):
#     """prepare batch data for training"""
#     # extract components
#     data = [item[0] for item in batch]
#     moves = [item[1] for item in batch]
#     board_states = [item[2] for item in batch]
#     labels = [item[3] for item in batch]

#     # validate board states
#     for i, bs in enumerate(board_states):
#         if len(bs) == 0:
#             print(f"Warning: board_state at index {i} is empty.")
#         if any(val is None for row in bs for val in row):
#             print(f"Warning: board_state at index {i} contains None values.")

#     lengths = torch.tensor([len(bs) for bs in board_states], dtype=torch.int32)

#     # pad sequences and convert to float32
#     data_padded = pad_sequence([torch.tensor(d, dtype=torch.float32) for d in data], batch_first=True, padding_value=0)
#     moves_padded = pad_sequence([torch.tensor(m, dtype=torch.float32) for m in moves], batch_first=True, padding_value=0)
#     board_states_padded = pad_sequence([bs.clone().detach().to(torch.float32) for bs in board_states], batch_first=True, padding_value=0)

#     # combine labels into tensor
#     labels_stacked = torch.stack([l.clone().detach() for l in labels])

#     return data_padded, moves_padded, board_states_padded, labels_stacked, lengths


# def get_data_loaders(game_dataset: ChessDataset, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, Subset]:

#     # split data into train/validation/test sets
#     test_size = int(0.2 * len(game_dataset))  # 20% of total
#     train_size = len(game_dataset) - test_size
#     train_data, test_data = random_split(game_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

#     validation_size = int(0.2 * train_size)  # 20% of leftover training (16% of total)
#     train_size_split = train_size - validation_size
#     train_set_split, validation_set_split = random_split(train_data, [train_size_split, validation_size], generator=torch.Generator().manual_seed(42))

#     # Organize into loaders for the model
#     train_loader = DataLoader(train_set_split, batch_size=batch_size, collate_fn=collate_fn)
#     validation_loader = DataLoader(validation_set_split, batch_size=batch_size, collate_fn=collate_fn)
#     test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

#     return train_loader, validation_loader, test_loader, train_set_split