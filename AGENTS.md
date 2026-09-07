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

## NPC Acceptance Video Requirements

When generating an NPC acceptance video, load the target office MJCF unchanged and use only its
existing scene lighting, headlight, environment, materials, and furniture. Do not replace the
office with a standalone presentation scene or add review-only lights.

- Keep the camera close enough to show the NPC's entire body while preserving enough margin to
  inspect its action animation, texture map, and frame-synchronised OBJ accessory detail.
- Include close front, side, and seated shots. Exercise the relevant animation frames rather than
  presenting a single static mesh.
- Treat occlusion, colour/material stability, alpha-frame switching (no flicker), and missing
  mesh/texture/anchor assets as explicit acceptance checks. Preserve MuJoCo/asset-loader errors;
  do not hide a failed load with a fallback model.
- Deliver the MP4 with a machine-readable acceptance report that records the scene, NPC, native
  lighting policy, each shot, and any failed check. Generated videos and reports remain local
  review artifacts unless the task explicitly requests committing them.

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

## Agent Git Isolation and Long-Lived Workstreams

NPC development uses two long-lived workstream branches, not one branch per
small task:

- `feat/npc/appearance-optimization` owns NPC models, materials, accessories,
  appearance assets, appearance configuration, and their focused tests/tools.
- `feat/npc/action-optimization` owns animation, locomotion, action execution,
  controller/runtime behavior, recording used to validate actions, and their
  focused tests/tools.

`main` remains the original-project baseline, and `feat/npc-system` remains the
NPC integration branch. Never commit or push directly to `main`; only the
designated integration owner may update `feat/npc-system`.

### Branch and worktree limits

- Do not create a branch or worktree for an individual fix, experiment, test,
  asset adjustment, or documentation update. Commit that work atomically to the
  applicable long-lived workstream branch.
- No additional `feat/npc/*`, `fix/npc/*`, `test/npc/*`, or `docs/npc/*` branch
  may be created unless a human explicitly approves the new long-lived boundary
  before creation. A convenient name, a collision, or an existing dirty
  worktree is not approval to create another branch.
- Use only the maintained sibling worktrees
  `../stretch_mujoco-npc-appearance-optimization` and
  `../stretch_mujoco-npc-action-optimization`. Do not run `git worktree add` for
  routine NPC work.
- Only one agent may write in a workstream worktree at a time. Before editing,
  confirm the branch, inspect `git status --short` and `git diff --name-only`,
  and stop if another agent or the user has uncommitted work there. Other agents
  may inspect the worktree read-only or wait for ownership to be handed off.
- Treat every pre-existing modification, untracked file, ignored file, and
  commit as user-owned. Never stash, reset, clean, overwrite, move, squash, or
  discard it merely to obtain a clean worktree.
- Configure only this repository's local commit identity as
  `kolp323 <1317416016@qq.com>`; never change the global Git identity.

The branch count represents durable development directions, not the number of
tasks or commits. Follow-up fixes stay on the same workstream branch. If a task
crosses both workstreams, assign ownership by its primary behavior and keep the
cross-stream portion minimal and explicit; do not create a third branch.

### Produce atomic, reviewable commits

An agent may create local commits without another prompt only while it owns the
applicable workstream. Each commit must represent one coherent change, include
its directly supporting tests, and leave the branch usable. Use Conventional
Commit subjects such as `feat(npc):`, `fix(npc):`, `test(npc):`, and
`docs(npc):`.

Before every commit:

```bash
git config --local user.name
git config --local user.email
git status --short
git diff -- <task-paths>
# Run the focused checks defined for this task.
git add <explicit-path-1> <explicit-path-2>
git diff --cached --check
git diff --cached --name-status
git diff --cached
git commit -m "<type>(npc): concise task summary"
git status --short
```

The identity must be `kolp323 <1317416016@qq.com>`. Stage explicit paths only;
never use `git add .`, `git add -A`, or `git commit -a`. Do not stage `.env`,
`*.local.json`, private SMPL-X inputs, unmanifested generated assets,
checkpoints, datasets, recordings, `outputs/`, `aaa_workspace/task/`, diagnostic
dumps, or unrelated user work.

Run focused checks that exercise the changed behavior. Run the full suite and
pre-commit when practical. Record exact commands and observed results, including
known environment-dependent failures. Never claim an unrun check passed. End
with a clean worktree and report the workstream branch, commit SHA, checks, and
remaining risks to the integration owner.

### Remote and pull-request policy

Workstream agents must not push branches or create, update, close, or otherwise
operate on pull requests. They must not use GitHub APIs or other remote mutation
mechanisms unless a human explicitly requests the exact remote action. GitHub
access being available is not authorization.

There are no workstream PRs. A PR may be created only after the entire NPC
module is ready for integration and a human explicitly requests it. At that
point, only the integration owner may update `feat/npc-system`, rebase it onto
`origin/main`, run the agreed integration checks, review the full integration
diff, and open the single final PR to `main`. Without that explicit human
request, leave all remote refs and PR state unchanged.

## Security & Configuration Tips

Never commit API credentials, `.env` files, `*.local.json`, private SMPL-X parameters, checkpoints, datasets, recordings, or generated outputs. Use ignored local templates such as `stretch_mujoco/models/office_llm.example.json`, and keep credential files restricted (for example, mode `600`).
