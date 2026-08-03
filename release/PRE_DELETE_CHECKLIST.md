# Local deletion checklist

Run the following before deleting a DynaExQ checkout or its local model
directories:

```bash
python scripts/pre_delete_audit.py
```

The audit verifies that:

1. the working tree has no non-ignored changes;
2. local `HEAD`, `origin/master`, and the live remote `master` agree;
3. every model in `release/model_registry.json` still exists at its immutable
   revision and each DynaExQ release has the expected weight-byte total;
4. a local ShareGPT workload, when present, has the pinned upstream SHA-256;
5. every ignored path belongs to a reviewed category.

## Deliberately excluded from the public release

The following local-only material is not required to rebuild the paper or run
the released experiments:

- Python, pytest, package, and LaTeX build caches;
- the duplicate `DynExq_paper/` assembly directory;
- obsolete ACM template files and generated text/PDF copies;
- execution logs whose machine-readable result JSON is already committed;
- obsolete or incomplete local model directories explicitly rejected by
  `release/model_registry.json`.

The audit treats internal planning notes and reviewer material differently.
Those files are not suitable for the public repository, but they may have
personal historical value. The command exits nonzero while they are present
unless `--allow-private-note-loss` is supplied. This flag does not delete
anything; it only records an explicit decision that the notes may be lost.

## Scope outside this release

Unrelated model directories such as `piiranha-v1` and
`granite-3.3-2b-instruct` are outside DynaExQ's preservation guarantee. Do not
delete a broad parent directory unless those unrelated assets have been
handled separately.
