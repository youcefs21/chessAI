import chess
from chess import pgn

import pandas as pd


input_file = "test_data.pgn"
input_pgn_file_path = f"OriginalData/{input_file}"
output_pgn_file_path = f"OriginalData/features_{input_file}".replace(".pgn", ".pkl")
ftr = [3600, 60, 1]


def move_to_tuple(move: pgn.ChildNode) -> tuple:
    split_comment = move.comment.split("] [")
    eval_c = None  # if no eval, will show up as NaN
    clk_c = None
    for c in split_comment:
        if "eval" in c:
            eval_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            if "#" in eval_c_s:
                mate_in = eval_c_s.split("#")[1].replace("-", "")
                eval_c = max(320 - int(mate_in) * 10, 40)  # if a side has mate in x moves, they automatically get a rating of 40

                if "-" in eval_c_s:
                    eval_c = -eval_c
            else:
                eval_c = float(eval_c_s)
        elif "clk" in c:
            clk_c_s = c.replace("]", "").replace("[", "").split(" ")[1]
            # Convert string formatted time to seconds
            clk_c = sum([a * b for a, b in zip(ftr, map(int, clk_c_s.split(":")))])

    return move.san(), move.uci(), eval_c, clk_c


game_ending_reasons = []
with open(input_pgn_file_path) as in_pgn:
    game = pgn.read_game(in_pgn)
    headers_to_keep = sorted(
        [
            # "Event",
            # "Site",
            # "Date",
            # "Round",
            # "White",
            # "Black",
            "Result",
            # "UTCDate",
            # "UTCTime",
            "WhiteElo",
            "BlackElo",
            # "WhiteRatingDiff",
            # "BlackRatingDiff",
            # "ECO",
            # "Opening",
            "TimeControl",
            "Termination",
        ]
    )

    # headers = list(game.headers)

    games = []

    while game is not None:

        game_headers: dict[str, str | int] = dict(filter(lambda elem: elem[0] in headers_to_keep, game.headers.items()))
        game_headers["BlackElo"] = int(game_headers["BlackElo"])
        game_headers["WhiteElo"] = int(game_headers["WhiteElo"])

        if game_headers["Result"] == "1-0":
            game_headers["Result"] = 1
        elif game_headers["Result"] == "0-1":
            game_headers["Result"] = -1
        else:
            game_headers["Result"] = 0

        if game_headers["Termination"] not in game_ending_reasons:
            game_ending_reasons.append(game_headers["Termination"])

        game_headers_values = list(
            map(
                lambda elem: elem[1],
                sorted(
                    list(game_headers.items()),
                    key=lambda elem: elem[0],
                ),
            )
        )

        # header_values = list(dict(game.headers).values())
        # if len(game_headers) != len(headers_to_keep):
        #     print(headers_to_keep)
        #     print(game_headers)
        #     print(dict(game.headers))

        moves = []
        first_move_player = 0
        for move in game.mainline():
            moves.append([first_move_player, *move_to_tuple(move)])
            first_move_player = (first_move_player + 1) % 2

        pd_moves = pd.DataFrame(moves, columns=["Player", "Move (SAN)", "Move (UCI)", "Eval", "Time"])
        # print(pd_moves)
        games.append([*game_headers_values, pd_moves])

        # break

        game = pgn.read_game(in_pgn)
    games_pd = pd.DataFrame(games, columns=headers_to_keep + ["Moves"])
    print(games_pd)
    games_pd.to_pickle(output_pgn_file_path)
    print("End of file")
    print("Game ending reasons:", game_ending_reasons)

print(pd.read_pickle(output_pgn_file_path))
