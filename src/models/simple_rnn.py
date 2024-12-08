# player (2), piece being moved (6), from square (64)
# that's an input size of only 136 when hotshot encoded!

# capture (1), piece captured (6)
# so 143 with these extra features

import logging
import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

PIECE_CHANNELS = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
}

ELO_SCALER = MinMaxScaler()

logger = logging.getLogger("chessAI")


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


def preprocess_data(data):
    # Normalize Elo ratings
    elo_data = data[['WhiteElo', 'BlackElo']].values
    normalized_elos = ELO_SCALER.fit_transform(elo_data)
    
    # Existing preprocessing
    players = data["Player"].apply(aint).to_numpy()
    moving_pieces = data["MovingPiece"].apply(encode_piece).to_numpy()
    captured_pieces = data["CapturedPiece"].apply(encode_piece).to_numpy()
    uci_moves = data["UCI"].apply(encode_ucis).to_numpy()
    results = data["Result"].to_numpy()

    # Create a list to store all games
    games = []

    # Iterate through each game's data
    for i in range(len(players)):
        # Create array of normalized Elos for this game
        game_elos = np.tile(normalized_elos[i], (len(players[i]), 1))
        
        # Concatenate all features including Elos
        game_features = np.concatenate([
            players[i],              # Player info
            moving_pieces[i],        # Moving piece info
            captured_pieces[i],      # Captured piece info
            uci_moves[i],           # UCI move encoding
            game_elos               # Normalized Elo ratings (2 values per move)
        ], axis=1)
        
        games.append(game_features)

    return games, results


class ChessRNN:
    def __init__(self, sequence_length=10):
        """
        Initialize the RNN model
        sequence_length: number of moves to consider from start of game
        """
        logger.info(f"Initializing ChessRNN with sequence length {sequence_length}")
        self.sequence_length = sequence_length
        self.model = None

    def build_model(self, input_shape):
        """
        Build the RNN model architecture
        input_shape: (sequence_length, features_per_move)
        """
        logger.info(f"Building model with input shape {input_shape}")
        self.model = Sequential([SimpleRNN(64, input_shape=input_shape, return_sequences=True), SimpleRNN(32), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])  # Binary classification (win/loss)

        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
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
                X.append(sequence)
                y.append(result)

        logger.info(f"Prepared {len(X)} sequences")
        return np.array(X), np.array(y)

    def train(self, games, results, validation_split=0.2, epochs=10, batch_size=32):
        """
        Train the model on the game sequences
        """
        logger.info("Starting model training...")
        # Prepare sequences
        X, y = self.prepare_sequences(games, results)

        # Build model if not already built
        if self.model is None:
            input_shape = (self.sequence_length, X.shape[2])
            self.build_model(input_shape)

        # Train the model
        logger.info(f"Training model with {len(X)} samples over {epochs} epochs")
        history = self.model.fit(X, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

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


def train_rnn(data, steps_per_game):
    logger.info(f"Training RNN with {steps_per_game} steps per game")
    rnn = ChessRNN(steps_per_game)
    games, results = preprocess_data(data)
    logger.info(f"Preprocessed {len(games)} games")
    history = rnn.train(games, results, epochs=200)
    logger.info("RNN training completed")
    return rnn
