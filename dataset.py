from huggingface_hub import HfApi

class ChessDataset:
    def __init__(self):
        if not HfApi().repo_exists("Youcef/chessData"):
            print("repo does not exist, creating...")
            HfApi().create_repo(repo_id="Youcef/chessData", repo_type="dataset")
        else:
            print("repo exists")
