from typing import Iterable
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch.nn as nn
import torch
import numpy as np
from chess import pgn
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# use a gpu if available, to speed things up
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("mps available")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# core game metadata we want to keep
HEADERS_TO_KEEP = sorted(
    [
        "Result",
        "WhiteElo",
        "BlackElo",
    ]
)

# columns for each move's data
MOVE_HEADER_NAMES = [
    "Player",
    "Eval",
    "Time",
    "Board State",
]

# mapping of piece types to channel indices for board representation
PIECE_CHANNELS = {
    "P": 1,  # white pawn
    "N": 2,  # white knight
    "B": 3,  # white bishop
    "R": 4,  # white rook
    "Q": 5,  # white queen
    "K": 6,  # white king
    "p": 7,  # black pawn
    "n": 8,  # black knight
    "b": 9,  # black bishop
    "r": 10,  # black rook
    "q": 11,  # black queen
    "k": 12,  # black king
}


def move_to_tuple(move: pgn.ChildNode) -> tuple[float | None, int | None, str]:
    """extracts evaluation score, remaining time, and board state from a move"""
    # time conversion factors for hours:mins:secs
    ftr = [3600, 60, 1]

    # parse comment for eval and clock time
    split_comment = move.comment.split("] [")

    eval_c: float | None = None
    clk_c: int | None = None
    for c in split_comment:
        # engine evaluation of the move
        if "eval" in c:
            eval_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            if "#" not in eval_c_s:
                eval_c = float(eval_c_s)
                continue

            # handle mate-in-x positions
            mate_in = eval_c_s.split("#")[1].replace("-", "")
            eval_c = max(320 - float(mate_in) * 10, 40)

            if "-" in eval_c_s:
                eval_c = -eval_c
            continue

        # remaining time on the clock
        if "clk" in c:
            clk_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            clk_c = sum([a * b for a, b in zip(ftr, map(int, clk_c_s.split(":")))])

    return eval_c, clk_c, move.board().fen()


# convert data to more usable features (do this on the fly)
def filter_headers(game: pgn.Game) -> dict[str, str | int]:
    """filters game headers and converts values to appropriate types

    input values:
    - result: win/loss/draw (1-0, 0-1, 1/2-1/2)
    - whiteElo: player rating
    - blackElo: player rating

    output values:
    - result: 2=white win, 1=draw, 0=black win
    - whiteElo: int
    - blackElo: int
    """
    game_headers: dict[str, str | int] = dict(filter(lambda elem: elem[0] in HEADERS_TO_KEEP, game.headers.items()))
    game_headers["BlackElo"] = int(game_headers["BlackElo"])
    game_headers["WhiteElo"] = int(game_headers["WhiteElo"])

    # convert chess result to numeric value
    if game_headers["Result"] == "1-0":
        game_headers["Result"] = 2  # white win = 2
    elif game_headers["Result"] == "0-1":
        game_headers["Result"] = 0  # black win = 0
    else:
        game_headers["Result"] = 1  # draw = 1

    return game_headers


def pgn_game_to_data(game: pgn.Game) -> list:
    """converts a chess game into a structured data format"""
    game_headers = filter_headers(game)
    game_headers_values = list(
        map(
            lambda elem: elem[1],
            sorted(
                list(game_headers.items()),
                key=lambda elem: elem[0],
            ),
        )
    )

    # extract move data from pgn format and convert to array-like dataframe
    moves = []
    first_move_player = 0
    for move in game.mainline():
        moves.append([first_move_player, *move_to_tuple(move)])
        first_move_player = (first_move_player + 1) % 2

    pd_moves = pd.DataFrame(moves, columns=MOVE_HEADER_NAMES)

    return [*game_headers_values, pd_moves]


def iterate_games(input_pgn_file_path: str, limit: int | None = None) -> Iterable[pgn.Game]:
    """yields games from a pgn file, up to optional limit"""
    with open(input_pgn_file_path) as in_pgn:
        game = pgn.read_game(in_pgn)
        count = 0

        while game is not None:
            if limit is not None and count >= limit:
                break
            yield game
            game = pgn.read_game(in_pgn)
            count += 1


def pgn_file_to_dataframe(input_pgn_file_path: str, limit: int = 10000) -> pd.DataFrame:
    """converts pgn file contents to pandas dataframe"""
    game_iter = iterate_games(input_pgn_file_path, limit)

    games = []
    print("Loading data from file. This may take a while...")
    for i, game in enumerate(game_iter):
        games.append(pgn_game_to_data(game))
        if (i + 1) % 1000 == 0:
            print(f"processed {i + 1} games")
    print("Data loaded.")

    game_data = pd.DataFrame(games, columns=HEADERS_TO_KEEP + ["Moves"])
    return game_data


def board_fen_to_image(board_fen: str):
    """converts chess board fen string to 12-channel tensor representation

    each channel represents positions of one piece type (e.g. white pawns)
    output shape: (12, 8, 8) where 12 = number of piece types"""
    pieces = board_fen.split(" ")[0]
    rows = pieces.split("/")
    board = np.zeros((12, 8, 8), dtype=np.float32)

    # fill board tensor with piece positions
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


class ChessDataset(Dataset):
    def __init__(self, game_data: pd.DataFrame, only_use_first_X_moves: int | None = None):
        """dataset for chess games with normalized features"""
        game_data = game_data.dropna()
        self.labels = torch.tensor(game_data["Result"].to_numpy(), dtype=torch.float32)

        # each game has a list of moves -> this function needs to run on each list of moves
        def normalize_moves(moves_df):
            # normalize evaluation scores
            moves_df = moves_df.fillna(0)
            scaler = StandardScaler()
            moves_df["Eval"] = scaler.fit_transform(moves_df["Eval"].values.reshape(-1, 1)).flatten()

            # normalize remaining time relative to starting time
            initial_time = moves_df["Time"].iloc[0]
            moves_df["Time"] = moves_df["Time"] / initial_time

            # convert board states to tensor representation
            moves_df["Board State"] = moves_df.apply(lambda row: board_fen_to_image(row["Board State"]), axis=1)
            return moves_df

        game_data["Moves"] = game_data.apply(lambda row: normalize_moves(row["Moves"]), axis=1)

        self.board_states = game_data.apply(lambda row: row["Moves"]["Board State"].to_list(), axis=1)
        self.moves = game_data.apply(lambda row: row["Moves"].drop(columns=["Board State"]).to_numpy(), axis=1).to_numpy()

        self.game_metadata = game_data.drop(columns=["Moves", "Result"]).to_numpy()

        # normalize game metadata features
        self.game_metadata = StandardScaler().fit_transform(self.game_metadata)

        self.move_limit = only_use_first_X_moves

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        board_state_tensor = torch.tensor(np.array(self.board_states[idx]), dtype=torch.float32)
        moves = self.moves[idx]

        # restrict to first X moves if needed
        if self.move_limit is not None:
            moves = moves[: self.move_limit]
            board_state_tensor = board_state_tensor[: self.move_limit]

        return self.game_metadata[idx], moves, board_state_tensor, self.labels[idx]


class ChessNN(nn.Module):
    def __init__(self):
        """neural network for chess game outcome prediction

        architecture:
        1. cnn processes board states
        2. lstm processes move sequences
        3. fully connected layers combine features for final prediction"""
        super().__init__()

        metadata_per_move = len(MOVE_HEADER_NAMES) - 1
        metadata_per_game = len(HEADERS_TO_KEEP) - 1

        # process board states with cnn
        self.board_cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        # project cnn features to lstm input size
        self.flatten_cnn_output = nn.Linear(128 * 2 * 2, 256)

        # process move sequences
        self.rnn = nn.LSTM(input_size=256 + metadata_per_move, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        # classification layers
        self.fc = nn.Sequential(nn.Linear(256 * 2, 256), nn.ReLU(), nn.Dropout(0.4))

        # combine with game metadata
        self.fc2 = nn.Sequential(nn.Linear(256 + metadata_per_game, 128), nn.ReLU(), nn.Dropout(0.4))

        # final prediction layer
        self.fc3 = nn.Linear(128, 3)

    def forward(self, game_metadata, moves, board_states, lengths):
        batch_size, seq_len, _, _, _ = board_states.size()

        # process board states
        cnn_out = self.board_cnn(board_states.view(-1, 12, 8, 8))
        cnn_out = cnn_out.view(batch_size * seq_len, -1)
        cnn_out = self.flatten_cnn_output(cnn_out)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)

        # combine cnn features with move data
        combined_features = torch.cat((cnn_out, moves), dim=2)

        # process move sequences
        packed_features = pack_padded_sequence(combined_features, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        packed_rnn_out, _ = self.rnn(packed_features)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)

        # get final states
        idx = (lengths - 1).clone().detach().to(device=rnn_out.device).unsqueeze(1).unsqueeze(2).expand(-1, 1, rnn_out.size(2)).long()
        last_rnn_out = rnn_out.gather(1, idx).squeeze(1)

        # final classification
        output = self.fc(last_rnn_out)
        output = torch.cat((output, game_metadata), dim=1)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


def test_loss(model: nn.Module, test_loader: DataLoader, loss_function: nn.modules.loss._Loss) -> float:
    """calculate average loss on test data"""
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, moves, board_states, target, lengths in test_loader:
            # move data to device
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            lengths = lengths.to(device).float()
            target = target.to(device).long()

            # set model to eval mode and calculate loss
            model.eval()
            output = model(data, moves, board_states, lengths)
            loss = loss_function(output, target.long())
            test_loss += loss.item()

    return test_loss / len(test_loader)


def predict(model: nn.Module, test_loader: DataLoader) -> list[int]:
    """generate predictions for test data"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for data, moves, board_states, target, lengths in test_loader:
            # move data to device
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            lengths = lengths.to(device).float()

            # set model to eval mode and make prediction
            model.eval()
            output = model(data, moves, board_states, lengths)
            _, predicted = torch.max(output.data, 1)
            predictions += predicted.tolist()

    return predictions


def flatten(xss):
    """flatten nested list"""
    return [x for xs in xss for x in xs]


def train(
    model: nn.Module,
    loss_function: nn.modules.loss._Loss,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epoch: int,
    learning_rate: float,
    print_every: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """train model and track metrics"""
    train_losses: list[float] = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    # setup optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # reduce learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )

    for e in range(epoch):
        model.train()
        current_epoch_train_loss = 0

        for i, (data, moves, board_states, target, lengths) in enumerate(train_loader):
            # move data to device
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            target = target.to(device).long()
            lengths = lengths.to(device).float()

            optimizer.zero_grad()
            output = model(data, moves, board_states, lengths)
            loss = loss_function(output, target.long())
            loss.backward()

            # clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_epoch_train_loss += loss.item()

        # calculate metrics
        avg_train_loss = current_epoch_train_loss / len(train_loader)
        avg_test_loss = test_loss(model, test_loader, loss_function=loss_function)

        # save metrics for graphing
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        train_accuracy.append(
            accuracy(
                predict(model, train_loader),
                flatten([label[3] for label in train_loader]),
            )
        )
        test_accuracy.append(
            accuracy(
                predict(model, test_loader),
                flatten([label[3] for label in test_loader]),
            )
        )

        # reduce learning rate if needed
        scheduler.step(avg_test_loss)

        if e % print_every == 0:
            print(f"epoch {e} training loss: {avg_train_loss}")
            print(f"epoch {e} testing loss: {avg_test_loss}")
            print(f"current learning rate: {optimizer.param_groups[0]['lr']}")
            print(f"train accuracy: {train_accuracy[-1]}")
            print(f"test accuracy: {test_accuracy[-1]}")

    # Replace the pickle save with torch.save for the state dict
    model_save_path = "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return train_losses, test_losses, train_accuracy, test_accuracy


def accuracy(Y_pred, Y_test) -> float:
    """calculate prediction accuracy"""
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)
    correct = np.sum(Y_pred == Y_test)
    total = len(Y_test)
    return correct / total if total > 0 else 0


def collate_fn(batch):
    """prepare batch data for training"""
    # extract components
    data = [item[0] for item in batch]
    moves = [item[1] for item in batch]
    board_states = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    # validate board states
    for i, bs in enumerate(board_states):
        if len(bs) == 0:
            print(f"Warning: board_state at index {i} is empty.")
        if any(val is None for row in bs for val in row):
            print(f"Warning: board_state at index {i} contains None values.")

    lengths = torch.tensor([len(bs) for bs in board_states], dtype=torch.int32)

    # pad sequences and convert to float32
    data_padded = pad_sequence([torch.tensor(d, dtype=torch.float32) for d in data], batch_first=True, padding_value=0)
    moves_padded = pad_sequence([torch.tensor(m, dtype=torch.float32) for m in moves], batch_first=True, padding_value=0)
    board_states_padded = pad_sequence([bs.clone().detach().to(torch.float32) for bs in board_states], batch_first=True, padding_value=0)

    # combine labels into tensor
    labels_stacked = torch.stack([l.clone().detach() for l in labels])

    return data_padded, moves_padded, board_states_padded, labels_stacked, lengths


def get_data_loaders(game_dataset: ChessDataset, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, Subset]:

    # split data into train/validation/test sets
    test_size = int(0.2 * len(game_dataset))  # 20% of total
    train_size = len(game_dataset) - test_size
    train_data, test_data = random_split(game_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    validation_size = int(0.2 * train_size)  # 20% of leftover training (16% of total)
    train_size_split = train_size - validation_size
    train_set_split, validation_set_split = random_split(train_data, [train_size_split, validation_size], generator=torch.Generator().manual_seed(42))

    # Organize into loaders for the model
    train_loader = DataLoader(train_set_split, batch_size=batch_size, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_set_split, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, validation_loader, test_loader, train_set_split


if __name__ == "__main__":
    # set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    print("torch cuda available:", torch.cuda.is_available())

    # load and preprocess data
    data_path = f"Data/2024-08/xaa.pgn"
    game_data = pgn_file_to_dataframe(data_path)
    game_dataset = ChessDataset(game_data, 10)

    # split data into train/validation/test sets
    train_loader, validation_loader, test_loader, train_set_split = get_data_loaders(game_dataset, batch_size=128)

    # initialize model
    model = ChessNN()
    model.to(device)

    # calculate class weights for balanced training
    class_counts = torch.bincount(torch.tensor([int(label) for label in game_dataset.labels[train_set_split.indices]]))
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    # initialize loss function with class weights
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    # train model
    train_losses, validation_losses, train_accuracy, validation_accuracy = train(
        model,
        loss_function=loss_function,
        train_loader=train_loader,
        test_loader=validation_loader,
        print_every=10,
        learning_rate=0.01,
        epoch=200,
    )
