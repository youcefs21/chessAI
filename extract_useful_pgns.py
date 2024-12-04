import chess
from chess import pgn
import pandas as pd

src_file = "lichess_db_standard_rated_2024-10.pgn"
input_pgn_file_path = f"OriginalData/{src_file}"
output_pgn_file_path = f"OriginalData/parsed_{src_file}"


dataframes = []

games_with_eval_and_comment = 0
total_games = 0
games_with_no_first_move = 0
with open(input_pgn_file_path) as in_pgn:
    game = pgn.read_game(in_pgn)
    while game is not None:
        total_games += 1

        first_move = game.next()
        if first_move is None:
            games_with_no_first_move += 1
            game = pgn.read_game(in_pgn)
            continue

        if "clk" not in first_move.comment or "eval" not in first_move.comment:
            game = pgn.read_game(in_pgn)
            continue

        games_with_eval_and_comment += 1
        # print(first_move.san(), first_move.comment)
        print(game, file=open(output_pgn_file_path, "a"), end="\n\n")

        # print(game.headers)
        # print(game.mainline_moves())
        game = pgn.read_game(in_pgn)
    print("End of file")
    print("Num of valuable games:", games_with_eval_and_comment)
    print("Num of total games:", total_games)
    print("Num of games with no first move:", games_with_no_first_move)

# def move_to_list(move: chess.Move) -> tuple:
#     return move.san(), move.comment
