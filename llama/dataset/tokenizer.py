import torch
import numpy as np
import os
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if (128002 + i) not in self.special_tokens.values()
            }
        )

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(
        self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()
    ):
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        tokens += self.model.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)

    def tokenize_doc(self, doc: dict, data_type: type = np.uint32) -> np.ndarray:
        """
        Tokenizes a single document and returns a numpy array of tokens cast to the specified data type.

        :param doc: A document with a 'text' field to tokenize.
        :type doc: dict
        :param data_type: The type to cast the numpy array to. Defaults to np.uint32.
        :type data_type: type
        :return: Tokenized document as a numpy array.
        :rtype: np.ndarray
        """
        # Start with an <|end_of_text|> token
        tokens = [self.special_tokens["<|end_of_text|>"]]

        # Tokenize the document text
        tokens.extend(self.encode(doc["text"]))

        # Convert tokens to a numpy array and cast it to the specified data type
        tokens_np = np.array(tokens, dtype=data_type)

        return tokens_np

    def tokenize_str(self, string: str) -> torch.Tensor:
        """
        Tokenizes a single string and returns a tensor of integer tokens.

        :param string: A string to tokenize.
        :type string: str
        :return: Tokenized string as a tensor of integers.
        :rtype: torch.Tensor
        """
        # Tokenize the string using the custom encode method
        encoded = self.encode(string)
        # Return as a tensor, adding an extra dimension to match expected shape
        return torch.tensor(encoded).unsqueeze(0)

    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """
        Decodes a tensor of tokens into a string.

        :param tokens: A tensor of tokens to decode.
        :type tokens: torch.Tensor
        :return: A decoded string.
        :rtype: str
        """
        # Flatten the tensor to a list of tokens
        flat_tokens = tokens.squeeze(0).tolist()
        # Decode the token list back to a string using the custom decode method
        return self.decode(flat_tokens)
