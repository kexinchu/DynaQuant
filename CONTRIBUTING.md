# Contributing to DynaExQ

Contributions are welcome through GitHub issues and pull requests.

## Development setup

Create a Python 3.10+ environment and install the development dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Keep pull requests focused and include tests for behavior changes. GPU and
checkpoint-dependent tests must be opt-in; the default test suite should stay
CPU-safe.

## Reproducible experiments

Do not commit model checkpoints, downloaded benchmark data, raw user prompts,
credentials, machine-specific paths, or generated logs. Experiment outputs
intended for review should include the command, dependency versions, random
seed, hardware identity, model revision, and dataset revision.

## Licensing

By contributing, you agree that your contribution is licensed under the
Apache License 2.0. Preserve upstream copyright and license headers in adapted
model implementation files.
