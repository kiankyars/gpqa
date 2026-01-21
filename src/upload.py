import sys
import os
from huggingface_hub import HfApi
api = HfApi()

def upload_results(file_path):
    api.create_repo("kyars/gpqa-results", exist_ok=True, repo_type="dataset")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="kyars/gpqa-results",
        repo_type="dataset",
    )

if __name__ == "__main__":
    upload_results(sys.argv[1])