# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ MuJoCo simulation stack.

- `stretch_mujoco/` contains the simulator, robot interfaces, datamodels, navigation, agents, humanoid/NPC code, and `models/` XML, mesh, and texture assets.
- `examples/` contains runnable demos; `tests/` contains pytest coverage.
- `strech_codex/src/strech_codex/` contains the optional MCP/Codex agent package, with tests in `strech_codex/tests/`.
- `docs/` contains tutorials and contributor/development notes; `third_party/` holds the Robocasa and Robosuite submodules.

Keep changes close to the owning module, and update both client/server sides when changing simulator commands or shared state.

## Build, Test, and Development Commands

Use `uv` from the repository root and keep the lockfile authoritative:

```bash
uv sync --extra dev                 # Install the development environment
uv run pytest -q                    # Run the test suite
uv run pytest -q tests/test_office_scene.py  # Run focused tests
uv run pre-commit run --all-files   # Run formatting, lint, typing, and file checks
uv run launch_sim                   # Launch the interactive simulator
uv run examples/office_scene.py --headless  # Run a headless smoke test
uv build                            # Build the distributable package
```

The main project expects sibling path `../openpi/packages/openpi-client`; provision it or change the dependency deliberately in a reviewed PR. Initialize submodules with `git submodule update --init` when needed.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints, `snake_case` for functions/modules, `PascalCase` for classes, and test names beginning with `test_`. Black is configured for 100-character lines; flake8 allows 120. Use isort with the Black profile and run mypy through pre-commit. Preserve existing enum, dataclass, and XML asset conventions.

## Testing Guidelines

Add or update pytest tests for behavior changes. Run focused tests while iterating, then run `uv run pytest -q`. Simulator, rendering, asset, and optional-model tests may require headless execution, extra dependencies, or local assets; state requirements and results in the PR. No separate coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use prefixes such as `feat:`, `fix:`, and `chore:`, although older history is mixed; use one focused subject. PRs should describe scope, verification commands/results, asset regeneration, and any XML, site, semantic-ID, or public API changes. Include screenshots or reproduction steps for viewer, scene, or visual changes, and request review from the relevant module owner.

## Security & Configuration Tips

Never commit API credentials, `.env` files, `*.local.json`, private SMPL-X parameters, checkpoints, datasets, recordings, or generated outputs. Use ignored local templates such as `stretch_mujoco/models/office_llm.example.json`, and keep credential files restricted (for example, mode `600`).
