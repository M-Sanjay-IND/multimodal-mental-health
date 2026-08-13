import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo


def deploy_to_hf_space(space_id: str, token: str = None):
    """Pushes project code, models, and artifacts to Hugging Face Spaces (Docker SDK)."""
    print(f"[INFO] Preparing deployment to Hugging Face Space: {space_id}")

    api = HfApi()

    # 1. Ensure space exists
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=token,
            exist_ok=True,
        )
        print(f"[OK] Space repo target verified: https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"[WARN] Repo creation/check notice: {e}")

    # 2. Upload workspace folder excluding dockerignored files
    ignore_patterns = [
        ".git/*",
        ".pytest_cache/*",
        "__pycache__/*",
        "*.pyc",
        "datasets/*",
        "data/*.parquet",
        "tests/*",
        "scratch/*",
        ".env",
    ]

    print("[INFO] Uploading project files to Hugging Face Spaces...")
    api.upload_folder(
        folder_path=".",
        repo_id=space_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        token=token,
    )

    print(f"[SUCCESS] Deployed successfully to https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Multimodal Mental Health API to Hugging Face Spaces")
    parser.add_argument("--space-id", type=str, help="Hugging Face Space ID (e.g. username/multimodal-mental-health)")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face User Access Token")

    args = parser.parse_args()

    if not args.space_id:
        print("[NOTICE] Usage: python scripts/deploy_hf.py --space-id 'username/space-name'")
        sys.exit(0)

    deploy_to_hf_space(args.space_id, args.token)
