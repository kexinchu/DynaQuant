#!/usr/bin/env python3
"""Build a deterministic calibration corpus from a pinned training split."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-103-raw-v1"
DEFAULT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(
    dataset,
    *,
    count: int,
    min_chars: int,
    max_chars: int,
    dataset_name: str = f"{DEFAULT_REPOSITORY}/{DEFAULT_CONFIG}",
) -> tuple[list[dict], int]:
    """Concatenate consecutive nonempty training rows into stable prompts."""
    if count < 128:
        raise ValueError("formal calibration requires at least 128 prompts")
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("invalid prompt-length bounds")
    records: list[dict] = []
    parts: list[str] = []
    source_start: int | None = None
    source_end = -1
    source_rows_consumed = 0

    def flush() -> None:
        nonlocal parts, source_start, source_end
        if source_start is None:
            return
        prompt = "\n".join(parts).strip()
        if len(prompt) >= min_chars:
            records.append(
                {
                    "dataset": (
                        dataset_name
                    ),
                    "split": "train",
                    "id": f"rows-{source_start}-{source_end}",
                    "prompt": prompt,
                    "source_row_start": source_start,
                    "source_row_end": source_end,
                }
            )
        parts = []
        source_start = None
        source_end = -1

    for index, item in enumerate(dataset):
        source_rows_consumed = index + 1
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if source_start is None:
            source_start = index
        candidate_length = sum(len(part) + 1 for part in parts) + len(text)
        if parts and candidate_length > max_chars:
            flush()
            if len(records) >= count:
                break
            source_start = index
        parts.append(text)
        source_end = index
        if sum(len(part) + 1 for part in parts) >= max_chars:
            flush()
            if len(records) >= count:
                break
    if len(records) < count:
        flush()
    if len(records) < count:
        raise ValueError(
            f"training split yielded only {len(records)} calibration prompts"
        )
    return records[:count], source_rows_consumed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--min-chars", type=int, default=16384)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (output, manifest_path):
        if not path.is_relative_to(ROOT):
            raise SystemExit("outputs must be inside the repository")
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing file: {path}")

    dataset = load_dataset(
        args.repository,
        args.config,
        split="train",
        revision=args.revision,
    )
    records, rows_consumed = build_records(
        dataset,
        count=args.count,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        dataset_name=f"{args.repository}/{args.config}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "independent_calibration_corpus",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": args.repository,
            "config": args.config,
            "split": "train",
            "revision": args.revision,
            "fingerprint": dataset._fingerprint,
            "source_row_count": len(dataset),
            "source_rows_consumed": rows_consumed,
        },
        "selection": {
            "method": "consecutive_nonempty_rows",
            "count": len(records),
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "uses_evaluation_test_split": False,
        },
        "output": {
            "path": str(output.relative_to(ROOT)),
            "sha256": _sha256(output),
        },
        "selected_ids": [record["id"] for record in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
