from typing import Optional

from chess.pgn import Game
from chess import pgn

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
    board = [[[0 for _ in range(8)] for _ in range(8)] for _ in range(12)]

    # Populate piece planes
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
                continue

            piece_index = PIECE_CHANNELS[char]
            board[piece_index - 1][i][j] = 1
            j += 1

    return board


def move_to_tuple(
    move: pgn.ChildNode,
) -> tuple[float, float | None, str | None, list[list[list[int]]]]:
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

    return clk_c / 60, eval_c, eval_c_s, board_fen_to_image(move.board().fen())


def preprocess_game(game: Game):
    # Get game headers
    result = 1 if game.headers.get("Result") == "1-0" else 0
    white_elo = int(game.headers.get("WhiteElo"))
    black_elo = int(game.headers.get("BlackElo"))

    # Initialize game data dictionary
    game_data = {"Result": result, "WhiteElo": white_elo, "BlackElo": black_elo, "Player": [], "Time": [], "Eval": [], "Raw Eval": [], "Board": []}

    first_move_player = 0
    for move in game.mainline():
        time, eval_c, raw_eval, board_state = move_to_tuple(move)

        # Append each value to its respective list in the dict
        game_data["Player"].append(first_move_player)
        game_data["Time"].append(time)
        game_data["Eval"].append(eval_c)
        game_data["Raw Eval"].append(raw_eval)
        game_data["Board"].append(board_state)

        first_move_player = (first_move_player + 1) % 2

    # Convert dict to list using HEADERS order
    return [game_data[header] for header in HEADERS]


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
        and "eval" in first_move.comment
    )
