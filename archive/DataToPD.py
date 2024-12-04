import pandas as pd
import chess.pgn

pgn_file_path = "Data/lichess_db_standard_rated_2015-05.pgn"

# Initialize lists to store data
features = []
targets = []

# Read and process games
with open(pgn_file_path) as pgn_file:
    i = 0
    while True:
        i += 1
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break  # End of file
        
        # Extract data from headers
        winner = game.headers["Result"]  # '1-0', '0-1', '1/2-1/2'
        if winner == "1-0":
            targets.append("white")
        elif winner == "0-1":
            targets.append("black")
        else:
            targets.append("draw")
        
        # Extract features (example: players' ratings and moves)
        # ToDo add all features
        white_rating = int(game.headers.get("WhiteElo", 0))
        black_rating = int(game.headers.get("BlackElo", 0))
        num_moves = len(list(game.mainline_moves()))  # Number of moves
        
        features.append([white_rating, black_rating, num_moves])
    print(i)

# Convert to a DataFrame
data = pd.DataFrame(features, columns=["WhiteElo", "BlackElo", "NumMoves"])
data["Winner"] = targets

# Save DataFrame as a Pickle file
data.to_pickle("Data/chess_games.pkl")
print("DataFrame saved as chess_games.pkl")