import feature_handlers as fh
import pandas as pd


def main(parent_dir: str, src_file: str) -> None:
    parsed_file_name = f"parsed_{src_file}"
    feature_file_name = f"features_{src_file}"
    parsed_path = f"{parent_dir}/{parsed_file_name}"
    feature_path = f"{parent_dir}/{feature_file_name}".replace(".pgn", ".pkl")

    game_data = []
    for game in fh.iterate_games(parsed_path):
        game_header_values, game_moves = fh.pgn_game_to_data(game)
        game_data.append([*game_header_values, game_moves])

    games = pd.DataFrame(game_data, columns=[fh.HEADERS_TO_KEEP, "Moves"])
    games.to_pickle(feature_path)


if __name__ == "__main__":
    parent_dir = "OriginalData"
    src_file = "lichess_db_standard_rated_2024-10.pgn"
    main(parent_dir, src_file)
