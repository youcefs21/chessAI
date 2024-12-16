# player (2), piece being moved (6), from square (64)
# capture (1), piece captured (6), eval score (1)
# that's an input size of 144 when hotshot encoded!

import logging
import keras
import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.regularizers import l2
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

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
    print("global_min: ", global_min)
    print("global_max: ", global_max)

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


def decode_sequence_element(element):
    """
    Decodes a single sequence element into human-readable format.

    The element contains:
    - Player (1 value)
    - Moving piece (6 values)
    - Captured piece (6 values)
    - UCI move (128 values - 64 for from square, 64 for to square)
    - Eval score (1 value)
    - Elo ratings (2 values - White and Black)
    """
    # Reverse mappings
    piece_types = {v: k for k, v in PIECE_CHANNELS.items()}
    files = "abcdefgh"
    ranks = "12345678"

    # Extract different parts of the element
    player = int(element[0])
    moving_piece_hot = element[1:7]
    captured_piece_hot = element[7:13]
    uci_hot = element[13:141]  # 128 values for UCI encoding
    eval_score = element[141]
    white_elo = element[142]
    black_elo = element[143]

    # Decode moving piece
    moving_piece_idx = moving_piece_hot.argmax()
    moving_piece = piece_types.get(moving_piece_idx, "None")

    # Decode captured piece
    captured_piece_idx = captured_piece_hot.argmax()
    captured_piece = piece_types.get(captured_piece_idx, "None")

    # Decode UCI move
    from_square_hot = uci_hot[:64]
    to_square_hot = uci_hot[64:128]

    def square_index_to_notation(idx):
        file_idx = idx % 8
        rank_idx = idx // 8
        return f"{files[file_idx]}{ranks[rank_idx]}"

    from_square = square_index_to_notation(from_square_hot.argmax())
    to_square = square_index_to_notation(to_square_hot.argmax())
    uci = f"{from_square}{to_square}"

    # Format the output
    output = (
        f"Player: {'White' if player == 0 else 'Black'}\n"
        f"Moving Piece: {moving_piece}\n"
        f"Captured Piece: {captured_piece}\n"
        f"Move (UCI): {uci}\n"
        f"Eval Score: {eval_score:.3f}\n"
        f"White Elo: {white_elo:.3f}\n"
        f"Black Elo: {black_elo:.3f}\n"
    )

    return output


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
        Build the RNN model architecture with improved stability
        """
        logger.info(f"Building model with input shape {input_shape}")

        # Create model with Input layer explicitly
        inputs = keras.Input(shape=input_shape)
        metadata_inputs = inputs[:, :, -2:]  # Elo ratings are the last 2 features
        metadata_inputs = metadata_inputs[:, 0, :]

        x = Dropout(0.1)(inputs[:, :, :-2])

        # First RNN layer
        # x = keras.layers.LSTM(
        x = SimpleRNN(
            128,
            return_sequences=True,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001),
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            activation="tanh",
        )(x)
        x = Dropout(0.2)(x)

        # Second RNN layer
        x = SimpleRNN(
            64,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001),
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            activation="tanh",
        )(x)
        x = Dropout(0.2)(x)

        # Dense layers
        x = keras.layers.concatenate([x, metadata_inputs])  # Combine RNN output with Elo ratings
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.1)(x)
        x = Dense(16, activation="relu", kernel_regularizer=l2(0.001))(x)
        outputs = Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # Use a more stable optimizer configuration
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
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
                print("result: ", result)
                print(decode_sequence_element(sequence[0]))
                X.append(sequence)
                y.append(result)

        logger.info(f"Prepared {len(X)} sequences")
        return np.array(X), np.array(y)

    def train(
        self,
        games,
        results,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        model_name="best_model.keras",
    ):
        """
        Train the model with improved training process
        """
        logger.info("Starting model training...")
        X, y = self.prepare_sequences(games, results)

        if self.model is None:
            input_shape = (self.sequence_length, X.shape[2])
            self.build_model(input_shape)

        # More sophisticated callbacks
        callbacks = [
            # Early stopping with longer patience
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
            ),
            # Reduce learning rate when plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1,
            ),
            # Model checkpoint with .keras extension
            keras.callbacks.ModelCheckpoint(
                model_name,  # Changed from .h5 to .keras
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Class weights to handle imbalanced data
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
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_batch_size=batch_size,
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


def train_rnn(data, steps_per_game, model_name="best_model.keras", plot=True):
    logger.info(f"Training RNN with {steps_per_game} steps per game")
    rnn = ChessRNN(steps_per_game)
    games, results = preprocess_data(data)
    logger.info(f"Preprocessed {len(games)} games")
    history = rnn.train(games, results, epochs=200, batch_size=128, model_name=model_name)
    if plot:
        plot_training_history(history)
    logger.info("RNN training completed")
    return rnn


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


def evaluate_model(model, data, title="Model Evaluation", plot=True):
    """
    Comprehensive evaluation of the model including metrics and visualizations
    """
    logger.info("Starting model evaluation...")

    # Preprocess test data and get valid indices
    games, true_labels = preprocess_data(data)
    X, y = model.prepare_sequences(games, true_labels)

    # Get predictions for valid sequences only
    predictions_prob = model.predict(X)
    predictions = (predictions_prob > 0.5).astype(int).flatten()
    y = y.flatten()

    # Now true_labels and predictions will have matching lengths
    conf_matrix = confusion_matrix(y, predictions)
    class_report = classification_report(y, predictions)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, predictions_prob.flatten())
    roc_auc = auc(fpr, tpr)

    if plot:
        # Create visualization
        plt.figure(figsize=(15, 5))

        # Plot 1: Confusion Matrix
        plt.subplot(131)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{title}\nConfusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Plot 2: ROC Curve
        plt.subplot(132)
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        # Plot 3: Prediction Distribution
        plt.subplot(133)
        sns.histplot(predictions_prob, bins=50)
        plt.title("Prediction Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.show()

    # Print classification report and metrics
    logger.info("\nClassification Report:\n" + class_report)
    logger.info(f"ROC AUC Score: {roc_auc:.3f}")

    # Calculate additional metrics
    accuracy = (predictions == y).mean()
    logger.info(f"Accuracy: {accuracy:.3f}")

    # Log number of samples used
    logger.info(f"Evaluated on {len(y)} samples (filtered from {len(true_labels)} total games)")

    return {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "predictions": predictions,
        "probabilities": predictions_prob.flatten(),
        "n_samples": len(y),
        "n_total_games": len(true_labels),
    }
