import os
import argparse
import multiprocessing as mp
from datasets import load_from_disk
from llama.dataset import Tokenizer, ShardManager

from huggingface_hub import login
from huggingface_hub import hf_hub_download

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")

login(token=access_token)

tokenizer_file_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",
    filename="original/tokenizer.model",
    local_dir="llama32-files",
)
tokenizer = Tokenizer(tokenizer_file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenizes documents using multiprocessing."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="tokenized_data",
        help="Local directory to store data.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="exquisiteweb",
        help="Root path to the dataset shards.",
    )
    parser.add_argument(
        "--remote_name", type=str, default="sample-10BT", help="Remote dataset name."
    )
    parser.add_argument(
        "--customize_data",
        action="store_true",
        help="Customize data before tokenization.",
    )
    parser.add_argument(
        "--shard_size", type=int, default=int(1e8), help="Number of tokens per shard."
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.1,
        help="Quality threshold for filtering lines.",
    )
    return parser.parse_args()


# Preprocess each row based on the 'text' and 'line_quality' keys
def preprocess_row(row, quality_threshold):
    text_lines = row["text"].splitlines()
    line_quality = row["line_quality"]

    # Filter lines based on the quality threshold
    filtered_lines = [
        line
        for line, quality in zip(text_lines, line_quality)
        if quality >= quality_threshold
    ]

    # Rebuild the text by joining the filtered lines with newline characters
    new_text = "\n".join(filtered_lines)

    # Replace the 'text' field with the new preprocessed text
    row["text"] = new_text
    return row


# Generator to stream and preprocess data from multiple shards
def stream_from_shards(root_path, quality_threshold):
    # Find all directories that start with 'shard_' in the root path
    shard_dirs = [
        os.path.join(root_path, d)
        for d in os.listdir(root_path)
        if d.startswith("shard_")
    ]

    # Stream rows from each shard directory
    for shard_dir in shard_dirs:
        dataset = load_from_disk(shard_dir)
        for row in dataset:
            # Preprocess the row and yield it
            yield preprocess_row(row, quality_threshold)


def main():
    args = parse_args()

    current_script_path = os.path.dirname(__file__)
    project_root = os.path.dirname(current_script_path)
    data_dir = os.path.join(project_root, "data")
    DATA_CACHE_DIR = os.path.join(data_dir, args.local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    if args.customize_data:
        fw = stream_from_shards(args.dataset_path, args.quality_threshold)
    else:
        fw = load_dataset(args.dataset_path, name=args.remote_name, split="train")

    # Initialize Tokenizer and ShardManager
    shard_manager = ShardManager(DATA_CACHE_DIR, args.shard_size)

    # Use multiprocessing to tokenize documents
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenizer.tokenize_doc, fw, chunksize=16):
            shard_manager.add_tokens(tokens)

    shard_manager.finalize()


if __name__ == "__main__":
    main()
