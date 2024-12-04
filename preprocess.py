import numpy as np

piece_indices = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def preprocess_board(fen, time_left, eval_score, max_time=60, eval_scale=10):
    """
    Preprocess a chess board to a 17-channel image.

    There are 17 planes:
    Piece Planes (0-11): Each plane corresponds to one type of piece (e.g., white pawn, black rook).
    Active Color Plane (12): Indicates the player to move (1 for white, -1 for black).
    Castling Rights Plane (13): Encodes castling availability using a normalized value between -1 and 1.
    En Passant Plane (14): Marks the square where en passant capture is possible.
    Time Left Plane (15): Normalized time remaining in the game.
    Evaluation Score Plane (16): Normalized Stockfish evaluation score.
    """
    pieces = fen.split(' ')[0]
    active_color = fen.split(' ')[1]
    castling = fen.split(' ')[2]
    en_passant = fen.split(' ')[3]
    rows = pieces.split('/')
    board = np.zeros((8, 8, 17), dtype=np.float32)
    
    # Populate piece planes
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    j += 1
            elif char.isalpha():
                piece_index = piece_indices[char]
                board[i, j, piece_index-1] = 1
                j += 1
    
    # Set active color plane
    if active_color == 'w':
        board[:, :, 12] = 1
    else:
        board[:, :, 12] = -1
    
    # Set castling rights plane
    castling_rights = 0
    if 'K' in castling:
        castling_rights |= 1  # White Kingside
    if 'Q' in castling:
        castling_rights |= 2  # White Queenside
    if 'k' in castling:
        castling_rights |= 4  # Black Kingside
    if 'q' in castling:
        castling_rights |= 8  # Black Queenside
    board[:, :, 13] = (castling_rights / 15.0) * 2 - 1  # Normalize to [-1,1]
    
    # Set en passant plane
    if en_passant != '-':
        col = ord(en_passant[0]) - ord('a')
        row = 8 - int(en_passant[1])
        board[row, col, 14] = 1
    
    # Set time left plane
    board[:, :, 15] = (time_left / max_time) * 2 - 1  # Normalize to [-1,1]
    
    # Set Stockfish evaluation score plane
    board[:, :, 16] = eval_score / eval_scale  # Normalize based on eval_scale
    
    return board