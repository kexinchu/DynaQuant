from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import IO, Any, Dict, Optional


@dataclass
class TelemetryEvent:
    ts: float
    kind: str
    payload: Dict[str, Any]


class TelemetrySink:
    """Lightweight JSONL telemetry writer."""

    def __init__(self, path: Optional[Path] = None, stream: Optional[IO[str]] = None) -> None:
        if path is None and stream is None:
            raise ValueError("Either path or stream must be provided.")
        self._path = path
        self._stream = stream

    def emit(self, kind: str, **payload: Any) -> None:
        event = TelemetryEvent(ts=time.time(), kind=kind, payload=payload)
        line = json.dumps(asdict(event), separators=(",", ":"))
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()
        else:
            assert self._path is not None
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


__all__ = ["TelemetryEvent", "TelemetrySink"]

