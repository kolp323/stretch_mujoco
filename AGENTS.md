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

## Current Implementation Record Maintenance Protocol

Treat `aaa_workspace/docs/current.md` as the maintained implementation record for the ongoing NPC-system migration. Read the relevant sections before changing NPC schemas, assets, animation, attachment, interactions, agent execution, simulator transport, recording, or compatibility paths, and update the document in the same change when behavior or migration status changes.

- Keep the document evidence-based. Describe what the repository actually implements, why the chosen boundary exists, and which limitations remain; do not present declared protocols, adapters, or planned work as completed physical behavior.
- Record material additions by owning module and explain their contract, data flow, completion condition, failure behavior, and compatibility effect. Include public API, schema, semantic-ID, XML/site, asset-generation, or recording-format changes when applicable.
- Preserve the system ownership rules documented there: Agents own intent, `NpcController` owns per-NPC execution and animation state, MuJoCo owns observed pose and attachment state, and `SemanticWorld` receives effects only after successful physical receipts and verification.
- Keep configuration and generated assets explicit. Production NPC assets must be manifest-backed and validated; preview assets must remain labeled preview. Never commit private SMPL-X inputs or ignored generated derivatives, and never silently substitute preview assets for production assets.
- Maintain one authoritative animation-graph definition for baker and runtime behavior. Missing clips must remain observable through fallback events, and marker- or attachment-dependent actions must not be converted back to duration-only success.
- Preserve command idempotency, per-NPC sequencing, deadlines, terminal receipts, cross-NPC attachment claims, ordered interaction barriers, and semantic-commit idempotency when extending the protocol.
- Treat legacy APIs as migration adapters only. Do not add new behavior to the `set_humanoid_*`, legacy naming, schema-v1, or snapshot-v1 paths. Before deleting an adapter, migrate repository callers, add equivalent tests, and document the compatibility break.
- Update existing statements instead of merely appending contradictory status. Historical milestones may remain when clearly labeled as historical; the newest section must identify superseded claims and the current source of truth.
- Report exact verification commands and observed results. Distinguish focused tests from the full `tests/` suite, note required environment workarounds such as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and state baseline or dependency failures without claiming they passed.
- Never place credentials, private asset locations, machine-specific secrets, generated recordings, or restricted asset contents in `current.md`. Repository-relative paths and reproducible commands are preferred.
- At handoff, ensure the document's completed work, remaining work, and recommended next step agree with the code and tests in the worktree.

## Commit & Pull Request Guidelines

Recent commits use prefixes such as `feat:`, `fix:`, and `chore:`, although older history is mixed; use one focused subject. PRs should describe scope, verification commands/results, asset regeneration, and any XML, site, semantic-ID, or public API changes. Include screenshots or reproduction steps for viewer, scene, or visual changes, and request review from the relevant module owner.

### Git Hygiene and PR Workflow

- Configure and retain this repository's commit identity as `kolp323 <1317416016@qq.com>`.
- Develop each independently reviewable change on a focused feature branch (for example, `feat/npc-system`); do not commit directly to the integration branch.
- Keep each commit focused and independently understandable. Use conventional prefixes such as `feat(npc):`, `fix(npc):`, `test(npc):`, and `docs(npc):`; separate code, asset/XML, test, and documentation changes when that improves reviewability.
- Before staging, inspect `git status` and `git diff`. Stage explicit paths rather than using `git add .`, and do not include unrelated changes, local task files, generated outputs, recordings, or diagnostic artifacts. The repository ignores `/aaa_workspace/task/`, `/outputs/`, and `/stretch_mujoco/recording/` for this reason.
- Before opening or updating a PR, rebase the feature branch onto its target branch, resolve conflicts locally, and run the focused tests for the changed behavior. Run the full suite and pre-commit when practical; otherwise state exactly which checks were run and why broader checks were not run.
- Review the final PR diff against the target branch. The PR description must state scope, non-goals, verification commands and results, compatibility or public-interface changes, asset/XML regeneration details, and screenshots or reproduction steps for scene or viewer changes.
- After a rebase of an already-pushed feature branch, use `git push --force-with-lease`, never an unguarded force push.

## Security & Configuration Tips

Never commit API credentials, `.env` files, `*.local.json`, private SMPL-X parameters, checkpoints, datasets, recordings, or generated outputs. Use ignored local templates such as `stretch_mujoco/models/office_llm.example.json`, and keep credential files restricted (for example, mode `600`).
