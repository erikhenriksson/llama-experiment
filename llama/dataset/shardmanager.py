import numpy as np
from tqdm import tqdm
import os


def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    """
    Writes the tokenized data to a file.

    :param filename: The path to the file where the data will be saved.
    :param tokens_np: Numpy array of tokenized data to save.
    """
    print(f"Writing {filename}")
    np.save(filename, tokens_np)


class ShardManager:
    """Manages the token shards for multiprocessing document tokenization."""

    def __init__(self, base_dir, shard_size):
        self.base_dir = base_dir
        self.shard_size = shard_size
        self.current_shard = np.empty((shard_size,), dtype=np.uint32)
        self.token_count = 0
        self.shard_index = 0
        self.progress_bar = None

    def add_tokens(self, tokens):
        """Add tokens to the current shard, writing to file if the shard is full."""
        while len(tokens) > 0:
            available_space = self.shard_size - self.token_count
            if len(tokens) < available_space:
                self._add_to_shard(tokens)
                break
            else:
                self._add_to_shard(tokens[:available_space])
                self._write_shard()
                tokens = tokens[available_space:]

    def _add_to_shard(self, tokens):
        """Add tokens to the shard and update the progress bar."""
        end_index = self.token_count + len(tokens)
        self.current_shard[self.token_count : end_index] = tokens
        self.token_count += len(tokens)
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.shard_size, desc=f"Shard {self.shard_index}"
            )
        self.progress_bar.update(len(tokens))

    def _write_shard(self):
        """Write the current shard to a file and prepare for the next shard."""

        split = "val" if self.shard_index == 0 else "train"
        filename = os.path.join(self.base_dir, f"{split}_{self.shard_index:06d}.npy")

        write_datafile(filename, self.current_shard[: self.token_count])

        self.shard_index += 1
        self.token_count = 0
        self.progress_bar.close()
        self.progress_bar = None

    def finalize(self):
        """Write any remaining tokens as the last shard."""
        if self.token_count != 0:
            self._write_shard()
