from huggingface_hub import HfApi


def upload_model_folder(folder_path: str, repo_id: str):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    upload_model_folder(
        folder_path="outputs/qwen-responsible-ai-qlora",
        repo_id="YOUR_USERNAME/qwen-responsible-ai-qlora",
    )
