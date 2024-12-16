import torch
import matplotlib.pyplot as plt
import pandas as pd

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


# Hyperparameters to search
LR = [0.001, 0.01]
epochs = [5, 10, 100] # early stopping after certiain point same result
batch_sizes = [32, 128]

results = []

for lr in LR:
    for epoch in epochs:
        for batch_size in batch_sizes:
            rnn = train_rnn(chess_df.df_train, steps_per_game=30, plot=False, LR=lr, epochs=epoch, batch_size=batch_size)
            
            # Evaluate the model
            metrics = evaluate_model(rnn, chess_df.df_test, "Chess RNN Evaluation", plot=False)
            
            # Append results to the list
            results.append({
                "Learning Rate": lr,
                "Epochs": epoch,
                "Batch Size": batch_size,
                "Accuracy": metrics["accuracy"]
            })

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Plot graphs
plt.figure(figsize=(15, 10))

# Accuracy vs. Learning Rate
plt.subplot(2, 2, 1)
for batch_size in batch_sizes:
    filtered = results_df[results_df["Batch Size"] == batch_size]
    for epoch in epochs:
        subset = filtered[filtered["Epochs"] == epoch]
        plt.plot(subset["Learning Rate"], subset["Accuracy"], marker="o", label=f"Batch={batch_size}, Epoch={epoch}")
plt.title("Accuracy vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()

# Accuracy vs. Epochs
plt.subplot(2, 2, 2)
for lr in LR:
    filtered = results_df[results_df["Learning Rate"] == lr]
    for batch_size in batch_sizes:
        subset = filtered[filtered["Batch Size"] == batch_size]
        plt.plot(subset["Epochs"], subset["Accuracy"], marker="o", label=f"LR={lr}, Batch={batch_size}")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Accuracy vs. Batch Size
plt.subplot(2, 2, 3)
for lr in LR:
    filtered = results_df[results_df["Learning Rate"] == lr]
    for epoch in epochs:
        subset = filtered[filtered["Epochs"] == epoch]
        plt.plot(subset["Batch Size"], subset["Accuracy"], marker="o", label=f"LR={lr}, Epoch={epoch}")
plt.title("Accuracy vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.legend()

# Overall Summary
plt.subplot(2, 2, 4)
for lr in LR:
    filtered = results_df[results_df["Learning Rate"] == lr]
    plt.scatter(filtered["Epochs"], filtered["Accuracy"], c=filtered["Batch Size"], cmap="viridis", label=f"LR={lr}")
plt.title("Accuracy Overview")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.colorbar(label="Batch Size")
plt.legend()

plt.tight_layout()
plt.show()