from huggingface_hub import HfApi
import logging

logger = logging.getLogger('chessAI')

class ChessDataset:
    def __init__(self):
        logger.info("Initializing ChessDataset")
        api = HfApi()
        repo_id = "Youcef/chessAI"
        
        if not api.repo_exists(repo_id):
            logger.info(f"Repository {repo_id} does not exist, creating...")
            # api.create_repo(repo_id=repo_id, repo_type="dataset")
            # logger.info(f"Successfully created repository {repo_id}")
        else:
            logger.info(f"Repository {repo_id} already exists")
