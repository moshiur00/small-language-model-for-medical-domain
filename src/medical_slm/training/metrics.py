"""Append-only structured metrics with optional TensorBoard mirroring."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlMetricLogger:
    """Write one durable JSON object per training or evaluation event."""

    def __init__(
        self,
        path: str | Path,
        *,
        flush_each_record: bool = True,
        fsync_each_record: bool = False,
        tensorboard_log_directory: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_each_record = flush_each_record
        self.fsync_each_record = fsync_each_record
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._tensorboard_writer: Any | None = None

        if tensorboard_log_directory is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                self._file.close()
                raise ImportError(
                    "TensorBoard logging requires the 'tensorboard' package."
                ) from error
            self._tensorboard_writer = SummaryWriter(
                log_dir=str(tensorboard_log_directory)
            )

    def log(
        self,
        event: str,
        *,
        update: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate, write and flush one structured metric event."""
        if not event:
            raise ValueError("event cannot be empty.")
        if update < 0:
            raise ValueError("update cannot be negative.")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "update": update,
            "metrics": dict(metrics),
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        self._file.write(encoded + "\n")
        if self.flush_each_record:
            self.flush()

        if self._tensorboard_writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self._tensorboard_writer.add_scalar(
                        f"{event}/{name}",
                        value,
                        update,
                    )
        return record

    def flush(self) -> None:
        """Flush buffered JSONL and TensorBoard data."""
        self._file.flush()
        if self.fsync_each_record:
            os.fsync(self._file.fileno())
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.flush()

    def close(self) -> None:
        """Flush and close all metric sinks."""
        if self._file.closed:
            return
        self.flush()
        self._file.close()
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()

    def __enter__(self) -> JsonlMetricLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
