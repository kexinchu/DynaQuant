# Paper result artifacts

This directory contains machine-readable evidence associated with the DynaExQ
manuscript.

- Files at the directory root are registered motivation artifacts and the
  corresponding claim manifest.
- `performance/` preserves the available formal performance grid and its smoke
  runs exactly as produced by the experiment environment.
- `audits/` contains point-in-time audit snapshots.

The JSON files are intentionally preserved without rewriting local paths or
provenance fields. Presence in this directory does not by itself make an
artifact a formal manuscript claim: `manifest.json` is the authoritative
registry, and `python scripts/audit_paper_results.py` is the validation gate.

Large routing traces remain as JSON because the registered manifest hashes
their exact bytes. Model checkpoints, downloaded benchmark datasets, prompts,
and execution logs are not stored in this repository.
