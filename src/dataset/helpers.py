from typing import Optional, Tuple

from chess.pgn import Game
import pandas as pd
from chess import pgn

MOVE_HEADER_NAMES = [
    "Player",
    "Time",
    "Board State",
]

HEADERS_TO_KEEP = ["Result", "WhiteElo", "BlackElo"]


def move_to_tuple(
    move: pgn.ChildNode,
) -> tuple[float, str]:
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
    clk_c: Optional[int] = None
    for c in split_comment:
        if "clk" in c:
            clk_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            # Convert string formatted time to seconds
            clk_c = sum([a * b for a, b in zip(ftr, map(int, clk_c_s.split(":")))])

    return clk_c / 60, move.board().fen()


def preprocess_game(game: Game):
    game_headers = [
        1 if game.headers.get("Result") == "1-0" else 0,
        int(game.headers.get("WhiteElo")),
        int(game.headers.get("BlackElo")),
    ]
    moves = []
    first_move_player = 0
    for move in game.mainline():
        moves.append([first_move_player, *move_to_tuple(move)])
        first_move_player = (first_move_player + 1) % 2

    pd_moves = pd.DataFrame(moves, columns=MOVE_HEADER_NAMES)

    return [*game_headers, pd_moves]


def is_valid_game(game: Game) -> bool:
    game_length = sum(1 for _ in game.mainline())
    first_move = game.next()
    time_parts = game.time_control().parts
    if len(time_parts) != 1:
        return False
    time = time_parts[0]

    return (
        # needs to be a nice normal game with 20-60 moves
        game.headers.get("Termination") == "Normal"
        and game.headers.get("Result") != "1/2-1/2"  # exclude draws
        and first_move is not None
        and 20 <= game_length <= 60
        # standard time rules 60+0 time rules
        and time.time == 60
        and time.increment == 0
        and time.delay == 0
        # make sure the clock for each turn is present
        and "clk" in first_move.comment
    )
