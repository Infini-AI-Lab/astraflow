---
orphan: true
---

# Contributing

## Development Expectations

- Keep changes scoped and reproducible.
- Prefer explicit typing and existing dataclass configs.
- Use structured logging and avoid ad-hoc prints.

## Documentation Workflow

- Add pages under `docs/en/`.
- Register new pages in the `docs/en/index.rst` toctree.

## Testing and Validation

- Run focused tests for modified modules.
- If cluster-dependent tests cannot run locally, state this explicitly.
- Keep pre-commit checks green before opening a PR.

## Commit Style

- Use Conventional Commit prefixes (for example `docs:`, `feat:`, `fix:`).
- Keep commit subjects concise and imperative.
