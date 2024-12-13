import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import models.performance_metrics as pm


# Use a GPU if available, to speed things up
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS available")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def flatten(xss):
    return [x for xs in xss for x in xs]


def test_loss(model: nn.Module, test_loader: DataLoader, loss_function: nn.modules.loss._Loss) -> float:
    """Get the current loss of the data in the test_loader"""

    # Set model to evaluation mode
    model.eval()
    test_loss = 0

    # No backpropagation calculations needed
    with torch.no_grad():
        for data, moves, board_states, target, lengths in test_loader:

            # Move data to GPU if applicable
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            lengths = lengths.to(device).float()
            target = target.to(device).long()

            model.eval()

            # Predict the data, get the loss based on the prediction
            output = model(data, moves, board_states, lengths)
            loss = loss_function(output, target.long())
            test_loss += loss.item()

    # Average calculated loss so training and testing losses can be compared without considering dataset size
    return test_loss / len(test_loader)


def predict(model: nn.Module, test_loader: DataLoader) -> list[int]:
    """Predict the labels of the data in the test_loader using the model"""

    # Set model to evaluation mode
    model.eval()
    predictions = []

    # No backpropagation calculations needed
    with torch.no_grad():
        for data, moves, board_states, target, lengths in test_loader:

            # Move data to GPU if applicable
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            lengths = lengths.to(device).float()  # Use lengths directly

            model.eval()
            # Predict the data
            output = model(data, moves, board_states, lengths)

            # Index of the highest value in the predicted tensor will be our chosen class
            _, predicted = torch.max(output.data, 1)
            predictions += predicted.tolist()

    return predictions

def train(
    model: nn.Module,
    loss_function: nn.modules.loss._Loss,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epoch: int,
    learning_rate: float = 0.0003,  # Reduced initial learning rate
    print_every: int = 100,
) -> tuple[list[float], list[float]]:
    train_losses: list[float] = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Increased weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,  # More gradual reduction
        patience=10,  # Wait longer before reducing
        verbose=True,
        min_lr=1e-6,  # Don't let LR get too small
    )

    for e in range(epoch):
        # Training loop remains the same...
        model.train()
        current_epoch_train_loss = 0

        for i, (data, moves, board_states, target, lengths) in enumerate(train_loader):
            # Move data to GPU if applicable
            data = data.to(device).float()
            moves = moves.to(device).float()
            board_states = board_states.to(device).float()
            target = target.to(device).long()
            lengths = lengths.to(device).float()

            # Zero the gradients to prevent interference from previous iterations
            optimizer.zero_grad()

            # Predict the data
            output = model(data, moves, board_states, lengths)

            # Calculate the loss
            loss = loss_function(output, target.long())

            # Backpropagate to update parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_epoch_train_loss += loss.item()

        # Calculate average losses for this epoch
        avg_train_loss = current_epoch_train_loss / len(train_loader)
        avg_test_loss = test_loss(model, test_loader, loss_function=loss_function)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracy.append(
            pm.accuracy(
                predict(model, train_loader),
                flatten([label[3] for label in train_loader]),
            )
        )
        test_accuracy.append(
            pm.accuracy(
                predict(model, test_loader),
                flatten([label[3] for label in test_loader]),
            )
        )

        # Step the scheduler based on validation loss
        scheduler.step(avg_test_loss)

        # For debug
        if e % print_every == 0:
            print(f"Epoch {e} training loss: {avg_train_loss}")
            print(f"Epoch {e} testing loss: {avg_test_loss}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            print(f"Train accuracy: {train_accuracy[-1]}")
            print(f"Test accuracy: {test_accuracy[-1]}")

    return train_losses, test_losses, train_accuracy, test_accuracy


def collate_fn(batch):
    # Extract individual components from the batch
    data = [item[0] for item in batch]
    moves = [item[1] for item in batch]
    board_states = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    # Check for zero values in board_states and replace them with a suitable value (e.g., 1)
    # This assumes the board state should not have 0 values and replacing 0 with 1 makes sense in your case
    # Check for zero values in board_states and replace them with a suitable value (e.g., 1)
    for i, bs in enumerate(board_states):
        if len(bs) == 0:
            print(f"Warning: board_state at index {i} is empty.")
        # Check if the elements in the board state are valid (no None or invalid data)
        if any(val is None for row in bs for val in row):
            print(f"Warning: board_state at index {i} contains None values.")

    lengths = torch.tensor([len(bs) for bs in board_states], dtype=torch.int32)

    # Pad sequences and ensure float32 dtype
    data_padded = pad_sequence([torch.tensor(d, dtype=torch.float32) for d in data], batch_first=True, padding_value=0)
    moves_padded = pad_sequence([torch.tensor(m, dtype=torch.float32) for m in moves], batch_first=True, padding_value=0)
    board_states_padded = pad_sequence([bs.clone().detach().to(torch.float32) for bs in board_states], batch_first=True, padding_value=0)

    # Stack labels into a single tensor
    labels_stacked = torch.stack([l.clone().detach() for l in labels])

    return data_padded, moves_padded, board_states_padded, labels_stacked, lengths