import torch
import matplotlib.pyplot as plt

from dataset.chess_dataframe import ChessDataFrame, Sizes

from models.simple_rnn import train_rnn, evaluate_model

size = Sizes.mid
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
chess_df = ChessDataFrame(size=size)


acc = []
dimensions = (1, 61, 2)
for i in range(*dimensions):
    rnn = train_rnn(chess_df.df_train, i, f"test_models/best_model_{i}_moves.keras", False)
    metrics = evaluate_model(rnn, chess_df.df_test, "Chess RNN Evaluation", False)
    acc.append(metrics["accuracy"])

# Plot accuracy vs number of trained moves
plt.figure(figsize=(10, 6))
plt.plot(range(*dimensions), acc, marker="o", label="Accuracy")
plt.title("Accuracy vs. Number of Trained Moves")
plt.xlabel("Number of Trained Moves")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("test_models/accuracy_vs_moves.png")
plt.show()

