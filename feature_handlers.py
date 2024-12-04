from chess import pgn, Board
import pandas as pd
import numpy as np

from typing import Iterable

HEADERS_TO_KEEP = [
    "Result",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
    "Termination",
]


# ### Remove unnecessary data (do this to files permanently to reduce space)
def is_useful_game(game: pgn.Game) -> bool:
    """
    Determines if a game is useful by checking if it has eval and clk data in the first move.
    """
    first_move = game.next()
    if first_move is None:
        return False

    return "clk" in first_move.comment and "eval" in first_move.comment


# ### Convert data to more usable features (do this on the fly)
def filter_headers(game: pgn.Game) -> dict[str, str | int]:
    """
    Filters out unnecessary headers from a game and returns a dictionary of the remaining headers.
    We can likely just one-hot encode the termination and time control data.

    If input contains:
    - Result (1-0, 0-1, 1/2-1/2)
    - WhiteElo (str)
    - BlackElo (str)

    Output contains:
    - Result (1, 0, -1) - -1 for black win
    - WhiteElo (int)
    - BlackElo (int)
    """

    game_headers: dict[str, str | int] = dict(filter(lambda elem: elem[0] in HEADERS_TO_KEEP, game.headers.items()))
    game_headers["BlackElo"] = int(game_headers["BlackElo"])
    game_headers["WhiteElo"] = int(game_headers["WhiteElo"])

    if game_headers["Result"] == "1-0":
        game_headers["Result"] = 1
    elif game_headers["Result"] == "0-1":
        game_headers["Result"] = -1
    else:
        game_headers["Result"] = 0

    return game_headers


def move_to_tuple(move: pgn.ChildNode) -> tuple[str, str, float | None, int | None, str]:
    """
    Converts move data to a tuple of the following:
    - SAN (Standard Algebraic Notation) of the move
    - UCI (Universal Chess Interface) of the move
    - Evaluation of the move (if available)
    - Clock time left for the player of the move (if available)
    """
    # For time conversions to seconds
    ftr = [3600, 60, 1]

    # Dissect the comment part of the move to extract eval and clk
    split_comment = move.comment.split("] [")

    # If no eval or clk data, will show up as None/NaN
    eval_c: float | None = None
    clk_c: int | None = None
    for c in split_comment:
        if "eval" in c:
            eval_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            if "#" not in eval_c_s:
                eval_c = float(eval_c_s)
                continue

            # If a side has mate in x moves, they automatically get a min. rating of 40
            # Otherwise, less moves till mate => higher advantage for that side
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

    return move.san(), move.uci(), eval_c, clk_c, move.board().fen()


def pgn_game_to_data(game: pgn.Game) -> tuple[list, pd.DataFrame]:
    """
    Converts a pgn game to a list of game headers and a DataFrame of moves.

    Input:
    - game: pgn.Game object

    Output:
    - game_header_values: value of headers specified by filter_headers (in order)
    - pd_moves: DataFrame of moves
    """
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

    moves = []
    first_move_player = 0
    for move in game.mainline():
        moves.append([first_move_player, *move_to_tuple(move)])
        first_move_player = (first_move_player + 1) % 2

    pd_moves = pd.DataFrame(moves, columns=["Player", "Move (SAN)", "Move (UCI)", "Eval", "Time", "Board State"])

    return game_headers_values, pd_moves


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
    board = np.zeros((8, 8, 12), dtype=np.float32)

    # Populate piece planes
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    j += 1
            elif char.isalpha():
                piece_index = PIECE_CHANNELS[char]
                board[i, j, piece_index - 1] = 1
                j += 1

    return board


def iterate_games(input_pgn_file_path: str) -> Iterable[pgn.Game]:
    """Iterate over games in a PGN file."""
    with open(input_pgn_file_path) as in_pgn:
        game = pgn.read_game(in_pgn)

        while game is not None:
            yield game

            game = pgn.read_game(in_pgn)
