#!/usr/bin/env python3
"""
Sort experts per layer by activation counts and emit ordered expert ID lists.

Example
-------
python scripts/sort_expert_activations.py \
    --input /workspace/DynaQuant/DynaQuant_New/activations/activation_qwen30b_mmlu_pro.json \
    --output /workspace/DynaQuant/DynaQuant_New/activations/activation_qwen30b_mmlu_pro_sorted.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

LOGGER = logging.getLogger("sort_expert_activations")

LayerEntry = Union[Mapping[str, object],
                   Sequence[object], int, Tuple[object, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the activation statistics JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the sorted expert ID lists (JSON).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, ...).",
    )
    return parser.parse_args()


def extract_expert_activation(entry: LayerEntry) -> Tuple[int, float]:
    """
    Normalise various entry representations to (expert_id, activation_value).
    """
    if isinstance(entry, dict):
        if "expert_id" in entry and "activations" in entry:
            try:
                expert_id = int(entry["expert_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid expert_id value: {entry['expert_id']!r}") from exc
            try:
                activation = float(entry["activations"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid activations value: {entry['activations']!r}") from exc
            return expert_id, activation
        if len(entry) == 1:
            key, value = next(iter(entry.items()))
            try:
                expert_id = int(key)
                activation = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid expert/activation pair: {entry!r}") from exc
            return expert_id, activation
        raise ValueError(f"Unsupported mapping entry format: {entry!r}")

    if isinstance(entry, (list, tuple)):
        if len(entry) != 2:
            raise ValueError(
                f"Expected (expert_id, activation) pair, got: {entry!r}")
        expert_id_raw, activation_raw = entry
        try:
            expert_id = int(expert_id_raw)
            activation = float(activation_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pair entry: {entry!r}") from exc
        return expert_id, activation

    if isinstance(entry, int):
        raise ValueError(
            f"Encountered bare integer entry {entry!r}. "
            "Expected expert/activation pairs."
        )

    raise ValueError(f"Unrecognised layer entry format: {entry!r}")


def sort_layer(entries: Iterable[LayerEntry]) -> List[int]:
    """
    Sort entries by activation value (descending) and return expert IDs.
    """
    pairs = [extract_expert_activation(item) for item in entries]
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [expert_id for expert_id, _ in pairs]


def process_file(input_path: Path) -> Dict[str, List[int]]:
    LOGGER.info("Loading activations from %s", input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "Activation JSON must contain an object with layer entries.")

    sorted_map: Dict[str, List[int]] = {}
    for layer_name, entries in data.items():
        if layer_name.startswith("_"):
            LOGGER.debug("Skipping non-layer key: %s", layer_name)
            continue
        if not isinstance(entries, list):
            raise ValueError(
                f"Layer '{layer_name}' entries must be a list; received {type(entries).__name__}."
            )
        LOGGER.debug(
            "Processing %s with %d expert entries",
            layer_name,
            len(entries),
        )
        sorted_map[layer_name] = sort_layer(entries)

    return sorted_map


def save_output(output_path: Path, payload: Dict[str, List[int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Saved sorted expert IDs to %s", output_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    try:
        sorted_map = process_file(input_path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Failed to process activation file: %s", exc)
        return 1

    save_output(output_path, sorted_map)
    print(json.dumps(sorted_map, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
