"""Binary shard writer for fixed-length token sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medical_slm.data.tokenization.manifest import create_file_artifact


SUPPORTED_DTYPES: dict[str, np.dtype[Any]] = {
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
}


def select_token_dtype(vocabulary_size: int, configured_dtype: str = "auto") -> np.dtype[Any]:
    """Choose a safe unsigned integer dtype for token IDs."""
    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be greater than zero.")

    if configured_dtype == "auto":
        if vocabulary_size <= np.iinfo(np.uint16).max + 1:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    try:
        dtype = SUPPORTED_DTYPES[configured_dtype]
    except KeyError as error:
        raise ValueError(
            f"Unsupported token dtype '{configured_dtype}'. "
            f"Choose from: auto, {', '.join(sorted(SUPPORTED_DTYPES))}."
        ) from error

    maximum_token_id = vocabulary_size - 1
    if maximum_token_id > np.iinfo(dtype).max:
        raise ValueError(
            f"dtype {dtype.name} cannot represent token ID {maximum_token_id}."
        )

    return dtype


class BinaryShardWriter:
    """Write packed samples to deterministic binary shard files.

    Each stored sample contains ``sequence_length + 1`` tokens. The dataset
    later returns the first ``sequence_length`` tokens as ``input_ids`` and
    the last ``sequence_length`` tokens as next-token ``labels``.
    """

    def __init__(
        self,
        *,
        output_directory: Path,
        sequence_length: int,
        sequences_per_shard: int,
        dtype: np.dtype[Any],
        file_prefix: str = "shard",
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least one.")
        if sequences_per_shard < 1:
            raise ValueError("sequences_per_shard must be at least one.")

        self.output_directory = output_directory
        self.sequence_length = sequence_length
        self.sample_width = sequence_length + 1
        self.sequences_per_shard = sequences_per_shard
        self.dtype = np.dtype(dtype)
        self.file_prefix = file_prefix

        self.output_directory.mkdir(parents=True, exist_ok=True)

        self._token_buffer: list[int] = []
        self._sample_buffer: list[list[int]] = []
        self._shard_index = 0
        self._sequences_written = 0
        self._tokens_written = 0
        self._shard_paths: list[Path] = []

    @property
    def sequences_written(self) -> int:
        return self._sequences_written

    @property
    def tokens_written(self) -> int:
        return self._tokens_written

    @property
    def buffered_tail_tokens(self) -> int:
        return len(self._token_buffer)

    @property
    def shard_paths(self) -> list[Path]:
        return list(self._shard_paths)

    def add_tokens(self, token_ids: list[int]) -> None:
        """Append token IDs and emit every complete fixed-width sample."""
        if not token_ids:
            return

        self._token_buffer.extend(token_ids)

        while len(self._token_buffer) >= self.sample_width:
            sample = self._token_buffer[: self.sample_width]
            del self._token_buffer[: self.sample_width]
            self._sample_buffer.append(sample)

            if len(self._sample_buffer) >= self.sequences_per_shard:
                self._flush_sample_buffer()

    def _flush_sample_buffer(self) -> None:
        """Write currently buffered complete samples to one shard."""
        if not self._sample_buffer:
            return

        array = np.asarray(self._sample_buffer, dtype=self.dtype)
        expected_shape = (len(self._sample_buffer), self.sample_width)

        if array.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected shard shape {array.shape}; expected {expected_shape}."
            )

        shard_path = self.output_directory / f"{self.file_prefix}_{self._shard_index:05d}.bin"
        array.tofile(shard_path)

        self._shard_paths.append(shard_path)
        self._sequences_written += array.shape[0]
        self._tokens_written += int(array.size)
        self._shard_index += 1
        self._sample_buffer.clear()

    def finalize(self) -> dict[str, Any]:
        """Flush complete samples and report the discarded incomplete tail."""
        self._flush_sample_buffer()

        discarded_tail_tokens = len(self._token_buffer)
        self._token_buffer.clear()

        shard_artifacts = [
            {
                **create_file_artifact(path, relative_to=self.output_directory.parent),
                "sequence_count": (
                    path.stat().st_size
                    // self.dtype.itemsize
                    // self.sample_width
                ),
                "token_count": path.stat().st_size // self.dtype.itemsize,
            }
            for path in self._shard_paths
        ]

        return {
            "sequence_length": self.sequence_length,
            "sample_width": self.sample_width,
            "sequences_per_shard": self.sequences_per_shard,
            "dtype": self.dtype.name,
            "sequences_written": self._sequences_written,
            "tokens_written": self._tokens_written,
            "discarded_tail_tokens": discarded_tail_tokens,
            "shards": shard_artifacts,
        }
