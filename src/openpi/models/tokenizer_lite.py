"""
Lightweight PaliGemma tokenizer for Pi0.5 inference — no JAX/Flax dependency.

Mirrors PaligemmaTokenizer.tokenize_high_low_prompt() from tokenizer.py.
"""
import logging
import string
from pathlib import Path

import numpy as np
import sentencepiece

logger = logging.getLogger(__name__)

TOKENIZER_PATH = str(Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model")


class PaligemmaTokenizerLite:
    """Minimal tokenizer for Pi0.5 inference — replicates PaligemmaTokenizer without JAX."""

    def __init__(self, max_len: int = 256, tokenizer_path: str = TOKENIZER_PATH):
        self._max_len = max_len
        self._tokenizer = sentencepiece.SentencePieceProcessor()
        self._tokenizer.Load(tokenizer_path)
        logger.info("Loaded PaliGemma tokenizer from %s (vocab=%d)", tokenizer_path, self._tokenizer.GetPieceSize())

    def detokenize(self, tokens: np.ndarray) -> str:
        """Decode token ids to string, stopping at EOS (id=1) or padding (id=0)."""
        valid = [int(t) for t in tokens if t not in (0, 1)]
        return self._tokenizer.decode(valid)

    def tokenize_high_low_prompt(
        self,
        high_prompt: str,
        low_prompt: str,
        state: np.ndarray | None = None,
        actions=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build Pi0.5 token sequence for inference (same format as training).

        Returns (tokens, mask, ar_mask, loss_mask, subtask_region_mask, action_region_mask).
        """
        cleaned_high = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        cleaned_low = low_prompt.lower().strip().replace("_", " ").replace("\n", " ")

        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
        else:
            # Fallback: empty state
            state_str = " ".join(["127"] * 32)  # mid-range

        # Normalize high prompt
        if cleaned_high and cleaned_high[-1] in string.punctuation:
            cleaned_high = cleaned_high[:-1]
        cleaned_high += "."

        # Segment 1: task + state (no loss)
        sub_prompt_1 = f"Task: {cleaned_high}; State: {state_str}; Subtask: "
        tokens_1 = self._tokenizer.encode(sub_prompt_1, add_bos=True)
        ar_mask = [True] * len(tokens_1)
        loss_mask = [False] * len(tokens_1)
        subtask_region_mask = [False] * len(tokens_1)
        action_region_mask = [False] * len(tokens_1)

        # Segment 2: low-level subtask (loss computed here during training)
        # For inference (extract_latent), low_prompt is "" so this is minimal
        if cleaned_low and cleaned_low[-1] in string.punctuation:
            cleaned_low = cleaned_low[:-1]
        if cleaned_low:
            cleaned_low += "."
        sub_prompt_2 = cleaned_low + ";\nAction: "
        tokens_2 = self._tokenizer.encode(sub_prompt_2) + [1]  # EOS token
        ar_mask += [True] * len(tokens_2)
        loss_mask += [True] * len(tokens_2)
        subtask_region_mask += [True] * len(tokens_2)
        action_region_mask += [False] * len(tokens_2)

        tokens = tokens_1 + tokens_2

        # Pad / truncate to max_len
        n = len(tokens)
        if n < self._max_len:
            pad = self._max_len - n
            tokens = tokens + [False] * pad
            mask = [True] * n + [False] * pad
            ar_mask = ar_mask + [False] * pad
            loss_mask = loss_mask + [False] * pad
            subtask_region_mask = subtask_region_mask + [False] * pad
            action_region_mask = action_region_mask + [False] * pad
        else:
            if n > self._max_len:
                logger.warning("Token length %d exceeds max_len %d, truncating", n, self._max_len)
            tokens = tokens[:self._max_len]
            mask = [True] * self._max_len
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]
            subtask_region_mask = subtask_region_mask[:self._max_len]
            action_region_mask = action_region_mask[:self._max_len]

        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(mask, dtype=bool),
            np.asarray(ar_mask, dtype=np.int32),
            np.asarray(loss_mask, dtype=bool),
            np.asarray(subtask_region_mask, dtype=bool),
            np.asarray(action_region_mask, dtype=bool),
        )
