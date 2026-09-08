import re
import string
import torch
from typing import List, Dict, Optional


class TextPreprocessor:
    """
    Standardized Text Preprocessor and Tokenizer for Text-to-Video models.
    Provides:
    - Text prompt cleaning & normalization
    - Deterministic tokenization & vocabulary indexing
    - Length truncation and padding
    - Vectorized PyTorch tensor generation
    """
    def __init__(self, max_length: int = 32, vocab_size: int = 10000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def clean_text(self, text: str) -> str:
        """Sanitizes and normalizes prompt string."""
        if not text:
            return ""
        # Lowercase
        t = text.lower().strip()
        # Replace newlines/tabs with space
        t = re.sub(r"[\r\n\t]+", " ", t)
        # Collapse multiple whitespace
        t = re.sub(r"\s+", " ", t)
        return t

    def tokenize(self, text: str) -> List[str]:
        """Splits prompt into normalized word tokens."""
        clean = self.clean_text(text)
        # Remove punctuation except hyphens in words
        tokens = re.findall(r"\b[\w'-]+\b", clean)
        return tokens

    def _hash_token(self, token: str) -> int:
        """Hashes token into deterministic vocabulary index [4, vocab_size - 1]."""
        h = 0
        for char in token:
            h = (h * 31 + ord(char)) % (self.vocab_size - 4)
        return h + 4

    def tokenize_to_ids(self, text: str) -> List[int]:
        """Converts prompt text to integer token IDs with BOS and EOS tokens."""
        words = self.tokenize(text)
        ids = [self.bos_token_id]
        for w in words[:self.max_length - 2]:
            ids.append(self._hash_token(w))
        ids.append(self.eos_token_id)

        # Pad to max_length
        while len(ids) < self.max_length:
            ids.append(self.pad_token_id)

        return ids[:self.max_length]

    def tokenize_to_tensor(self, text: str) -> torch.Tensor:
        """Converts prompt directly to a 1D PyTorch LongTensor of length max_length."""
        ids = self.tokenize_to_ids(text)
        return torch.tensor(ids, dtype=torch.long)
