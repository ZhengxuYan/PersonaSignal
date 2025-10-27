import os
import argparse
import pandas as pd
from datasets import Dataset


def push_to_hf(csv_path, repo_id, token=None, private=False):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    dataset = Dataset.from_pandas(df)

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HF token found. Set HF_TOKEN env var or pass --token")

    print(f"Pushing to {repo_id}...")
    dataset.push_to_hub(repo_id, token=token, private=private)
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="username/dataset-name")
    parser.add_argument("--csv-path", default="data_with_personas.csv")
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")

    args = parser.parse_args()
    push_to_hf(args.csv_path, args.repo_id, args.token, args.private)
