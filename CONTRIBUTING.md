# Contributing

Thanks for your interest in densekit! It's a small project, so the process is
light.

## Development setup

```bash
git clone https://github.com/ivyw960/densekit
cd densekit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a PR

Run the same checks CI runs:

```bash
ruff check densekit tests
ruff format --check densekit tests
mypy
pytest --cov=densekit
```

All four must pass. `pre-commit` runs the lint/format hooks automatically on
commit.

## Guidelines

- **Keep the core NumPy-only.** New runtime dependencies in the core are a hard
  no; anything heavier (PyTorch, etc.) goes behind an optional extra and is
  imported lazily.
- **Stay offline-first.** No network calls or model downloads in the core or its
  tests.
- **Small, focused commits** with a test for each behaviour change.
- **Type everything.** Public functions carry annotations; `mypy` is enforced.
- Follow the existing style — `ruff format` decides formatting.

## Reporting bugs / requesting features

Use the issue templates. A minimal reproduction (a few lines of NumPy) makes
bugs far easier to fix.
