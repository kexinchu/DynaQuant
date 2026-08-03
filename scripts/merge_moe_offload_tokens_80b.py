#!/usr/bin/env python3
"""Merge partial 80B results (1..384) with tail run (512..1024) into complete moe_offload_tokens_qwen80b.json."""
import json
from pathlib import Path

results_dir = Path(__file__).resolve().parent / "results"
path_80b = results_dir / "moe_offload_tokens_qwen80b.json"
path_tail = results_dir / "moe_offload_tokens_80b_tail.json"
path_combined = results_dir / "moe_offload_tokens.json"

def main():
    with path_80b.open() as f:
        data_80b = json.load(f)
    if not path_tail.exists():
        print("Tail file not found, skipping merge.")
        return
    with path_tail.open() as f:
        tail = json.load(f)
    existing = {r["num_tokens"]: r for r in data_80b["results"]}
    for r in tail["results"]:
        existing[r["num_tokens"]] = r
    data_80b["results"] = [existing[n] for n in sorted(existing.keys())]
    with path_80b.open("w") as f:
        json.dump(data_80b, f, indent=2)
    print("Merged 80B results:", len(data_80b["results"]), "num_tokens")
    # Update combined if exists
    if path_combined.exists():
        with path_combined.open() as f:
            combined = json.load(f)
        models = combined.get("models", [])
        for i, m in enumerate(models):
            if "80B" in m.get("model", ""):
                models[i] = data_80b
                break
        combined["models"] = models
        with path_combined.open("w") as f:
            json.dump(combined, f, indent=2)
        print("Updated", path_combined)

if __name__ == "__main__":
    main()
