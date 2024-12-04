import numpy as np

piece_indices = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def preprocess_board(fen):
    """
    Preprocess a chess board to a 17-channel image.

    There are 17 planes:
    Piece Planes (0-11): Each plane corresponds to one type of piece (e.g., white pawn, black rook).
    """
    pieces = fen.split(' ')[0]
    rows = pieces.split('/')
    board = np.zeros((8, 8, 12), dtype=np.float32)
    
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
    
    return board