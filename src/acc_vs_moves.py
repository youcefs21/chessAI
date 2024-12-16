import torch
import matplotlib.pyplot as plt

from dataset.chess_dataframe import ChessDataFrame, Sizes

from models.simple_rnn import train_rnn, evaluate_model

size = Sizes.xxs
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
chess_df = ChessDataFrame(size=size)


acc = []

for i in range(10, 20):
    rnn = train_rnn(chess_df.df_train, i, False)
    metrics = evaluate_model(rnn, chess_df.df_test, "Chess RNN Evaluation", False)
    acc.append(metrics["accuracy"])

# Plot accuracy vs number of trained moves
plt.figure(figsize=(10, 6))
plt.plot(range(10, 20), acc, marker='o', label='Accuracy')
plt.title("Accuracy vs. Number of Trained Moves")
plt.xlabel("Number of Trained Moves")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()