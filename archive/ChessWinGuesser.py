import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from sklearn.model_selection import train_test_split 

class RNN:
    def __init__():
        return 

    def forward(self, x):
        # ToDo
        return

    def pre_process(self, X, Y):
        # ToDo
        return X, Y

    def train(self, X, Y):
        # Split the dataset into training and validation sets
        X_train, X_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Combine inputs and labels for training and validation
        train_data = list(zip(X_train, y_train))
        val_data = list(zip(X_validate, y_validate))

        # Create DataLoaders
        train_loader = dataloader.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = dataloader.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        # Set up optimizer and loss_function
        # optimizer = ToDo
        # loss_function = ToDO

        train_losses = []
        val_losses = []

        # for epoch in range(self.epoch):
        #     # Training loop
        #     epoch_loss = 0
        #     self.model.train()  # Set model to training mode
        #     for x, y in train_loader:
        #         output = self.forward(x)
        #         loss = loss_function(output, y)
        #         epoch_loss += loss.item()
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #     train_losses.append(epoch_loss / len(train_loader))

        #     # Validation loop
        #     val_loss = 0
        #     self.model.eval()  # Set model to evaluation mode
        #     with torch.no_grad():
        #         for x, y  in val_loader:
        #             output = self.forward(x)
        #             loss = loss_function(output, y)
        #             val_loss += loss.item()

        #     val_losses.append(val_loss / len(val_loader))

        #     print(f"Epoch {epoch+1}/{self.epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        return train_losses, val_losses

    def evaluate(self, X_test, Y_test):
        # ToDo
        return

    def predict(self, X):
        # toDO
        return

    def plot_losses(self, train_losses, val_losses):
        plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
        plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over Epochs")
        plt.show()

print("Loading data")
data = pd.read_pickle("Data/chess_games.pkl")
print("DataFrame loaded from chess_games.pkl")

# Split data into features (X) and target (y)
X = data[["WhiteElo", "BlackElo", "NumMoves"]]
y = data["Winner"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print summary
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# my_rnn = RNN()

# # pre process data
# X_train, y_train = my_rnn.pre_process(X_train, y_train)
# X_test, y_test = my_rnn.pre_process(X_test, y_test)

# train_losses, val_losses = my_rnn.train(X_train,y_train)
# my_rnn.plot_losses(train_losses, val_losses)
# acc = my_rnn.evaluate(X_test, y_test)