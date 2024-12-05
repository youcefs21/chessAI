import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

import feature_handlers as fh
import model_lib as ml


def main(parent_dir: str, src_file: str) -> None:
    # parsed_file_name = f"parsed_{src_file}"
    # feature_file_name = f"features_{src_file}"
    # parsed_path = f"{parent_dir}/{parsed_file_name}"
    # feature_path = f"{parent_dir}/{feature_file_name}".replace(".pgn", ".pkl")

    # game_data = []
    # for game in fh.iterate_games(parsed_path):
    #     game_header_values, game_moves = fh.pgn_game_to_data(game)
    #     game_data.append([*game_header_values, game_moves])

    # games = pd.DataFrame(game_data, columns=[fh.HEADERS_TO_KEEP, "Moves"])
    # games.to_pickle(feature_path)
    # Load the data for training and testing

    temp_test_path = f"Data/2024-08/xaa.pgn"
    game_data = fh.pgn_file_to_dataframe(temp_test_path)
    game_dataset = ml.ChessDataset(game_data)

    # print(len(game_dataset[0]))
    # print(game_dataset[0][0].dtype)
    # print(game_dataset[0][1].dtype)
    # print(game_dataset[0][2].dtype)
    # print(game_dataset[0][3].dtype)

    # train_set = load_mnist(parent_path, "train")
    # test_set = load_mnist(parent_path, "t10k")

    # Split the train set so we have a held-out validation set
    train_set_split, validation_set_split = random_split(game_dataset, [2, 1])

    # Initialize the model and move it to the GPU if available
    model = ml.ChessNN()
    model.to(ml.device)

    # Initialize the data loaders, using a batch size of 128
    batch_size = 1
    train_loader = DataLoader(train_set_split, batch_size=batch_size)
    validation_loader = DataLoader(validation_set_split, batch_size=batch_size)
    # test_loader = DataLoader(test_set, batch_size=128)

    # Train the model
    train_losses, validation_losses = ml.train(
        model,
        loss_function=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        test_loader=validation_loader,
        epoch=10,
        learning_rate=0.01,
    )


if __name__ == "__main__":
    parent_dir = "OriginalData"
    src_file = "lichess_db_standard_rated_2024-10.pgn"
    main(parent_dir, src_file)
