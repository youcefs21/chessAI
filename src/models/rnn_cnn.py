import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Use a GPU if available, to speed things up
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS available")
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class ChessNN(nn.Module):
    def __init__(self):
        super().__init__()

        metadata_per_move = 3 - 1
        metadata_per_game = 3 - 1

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

        self.flatten_cnn_output = nn.Linear(128 * 2 * 2, 256)

        self.rnn = nn.LSTM(input_size=256 + metadata_per_move, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.fc = nn.Sequential(nn.Linear(256 * 2, 256), nn.ReLU(), nn.Dropout(0.4))  # *2 because bidirectional

        self.fc2 = nn.Sequential(nn.Linear(256 + metadata_per_game, 128), nn.ReLU(), nn.Dropout(0.4))

        self.fc3 = nn.Linear(128, 3)

    def forward(self, game_metadata, moves, board_states, lengths):
        # Get batch and sequence dimensions
        batch_size, seq_len, _, _, _ = board_states.size()

        # Pass board states through CNN
        cnn_out = self.board_cnn(board_states.view(-1, 12, 8, 8))  # (batch_size * seq_len, cnn_output_dim)
        cnn_out = cnn_out.view(batch_size * seq_len, -1)  # Flatten for the fully connected layer

        # Pass through the fully connected layer to flatten
        cnn_out = self.flatten_cnn_output(cnn_out)  # (batch_size * seq_len, 128)

        # Reshape back to (batch_size, seq_len, -1) for RNN
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 128)

        # Combine CNN output with moves
        combined_features = torch.cat((cnn_out, moves), dim=2)  # (batch_size, seq_len, 128 + metadata_per_move)

        # Pack sequences for RNN
        packed_features = pack_padded_sequence(combined_features, lengths.to("cpu"), batch_first=True, enforce_sorted=False)

        # Pass through RNN
        packed_rnn_out, _ = self.rnn(packed_features)

        # Unpack the RNN output
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)  # (batch_size, seq_len, rnn_hidden_dim)

        # Use the last valid RNN output for each sequence
        idx = (lengths - 1).clone().detach().to(device=rnn_out.device).unsqueeze(1).unsqueeze(2).expand(-1, 1, rnn_out.size(2)).long()
        last_rnn_out = rnn_out.gather(1, idx).squeeze(1)  # (batch_size, rnn_hidden_dim)

        # Final prediction using fully connected layer
        output = self.fc(last_rnn_out)  # (batch_size, output_dim)
        output = torch.cat((output, game_metadata), dim=1)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
