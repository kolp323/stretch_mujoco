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

## Agent Git Isolation and Local Commits

Every agent-authored change must be developed as an independently reviewable
unit. A change task is complete only when it ends in at least one verified local
commit on a dedicated task branch, not as an uncommitted patch in a shared
worktree. These rules apply to code, tests, documentation, XML, and assets.

### Branch topology and ownership

- `main` is the original-project baseline. Never commit or push directly to it.
- `feat/npc-system` is the NPC integration branch and its final PR targets
  `main`. Only the designated integration owner may update this branch.
- Each task branches from `origin/feat/npc-system` as
  `feat/npc/<scope>`, `fix/npc/<scope>`, `test/npc/<scope>`, or
  `docs/npc/<scope>`. Task branches are local delivery units; they do not open
  task PRs.
- One agent owns one task branch and one worktree. Never share a task worktree,
  commit on another agent's branch, or mix independently reviewable tasks on a
  single branch.
- Configure only this repository's local commit identity as
  `kolp323 <1317416016@qq.com>`; never change the global Git identity.

### Start every task in an isolated worktree

Before editing, inspect the launching worktree:

```bash
git status --short
git diff --name-only
git branch --show-current
git fetch origin --prune
```

Treat every pre-existing modification and untracked file as user-owned. Do not
stash, reset, clean, overwrite, move, or include it in a task. A dirty launching
worktree must remain untouched, but it does not block creating a new sibling
worktree from the remote integration baseline:

```bash
task_type=feat
task_scope=short-unique-scope
git worktree add "../stretch_mujoco-npc-${task_scope}" \
  -b "${task_type}/npc/${task_scope}" origin/feat/npc-system
cd "../stretch_mujoco-npc-${task_scope}"
git status --short
git branch --show-current
```

The new task worktree must begin clean and on the intended task branch. If the
branch or worktree path already exists, choose a unique scope; do not reuse,
delete, or take ownership of an existing worktree. If isolation cannot be
established, stop before editing and report the exact conflict.

Before implementation, define the task boundary in working notes: one
objective, the expected files or owning modules, explicit non-goals, and the
focused verification command. Expand that boundary only when required for
correctness, and report why. Do not opportunistically fix unrelated issues.

### Produce atomic, reviewable commits

An agent may create local commits without another prompt when working on its
dedicated task branch. Each commit must represent one coherent change, be
independently understandable, and leave the branch in a usable state. Keep
implementation and its directly supporting tests together when they form one
behavioral unit; split unrelated documentation, generated assets/XML, or
mechanical changes when separate review or rollback would be clearer.

Use Conventional Commit subjects such as `feat(npc):`, `fix(npc):`,
`test(npc):`, and `docs(npc):`. Before every commit:

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
never use `git add .`, `git add -A`, or `git commit -a`. Before committing,
confirm every staged path is inside the declared task boundary and every
staged hunk is necessary. Do not stage `.env`, `*.local.json`, private SMPL-X
inputs, unmanifested generated assets, checkpoints, datasets, recordings,
`outputs/`, `aaa_workspace/task/`, diagnostic dumps, or unrelated user work.
Ignored files are still user-owned and are never disposable by default.

Run focused checks that exercise the changed behavior before committing. Run
the full suite and pre-commit when practical. A known or environment-dependent
failure may be committed only when the task change itself is complete, the
failure is not caused or hidden by the change, and the exact command and output
are recorded for handoff. Never claim an unrun check passed.

After the final commit, the task worktree should be clean. Any intentional
uncommitted file must be within task scope and explicitly reported; unrelated
or unexplained residue means the task is not ready for integration handoff.

### Prepare a clean local handoff

Synchronize and review only from a clean task worktree:

```bash
git status --short  # Must be empty before rebasing.
git fetch origin --prune
git rebase origin/feat/npc-system
# Rerun the relevant checks after rebasing.
git diff --check origin/feat/npc-system...HEAD
git diff --name-status origin/feat/npc-system...HEAD
git log --oneline origin/feat/npc-system..HEAD
git status --short
```

Review the complete range diff, not only the last commit. It must contain only
the declared task, with no merge commits, unrelated formatting, local files,
or generated artifacts. A completed task remains as verified local commits for
the integration owner to review and integrate.

Task agents must not push branches or create, update, close, or otherwise
operate on pull requests. They must not use GitHub APIs or other remote mutation
mechanisms unless a human explicitly requests the exact remote action. GitHub
access being available is not authorization. Do not report a push or PR command
as a required next step for an ordinary task handoff.

There are no task PRs. A PR may be created only after the entire NPC module is
ready for integration and a human explicitly requests it. At that point, only
the integration owner may rebase `feat/npc-system` onto `origin/main`, run the
agreed integration checks, review the full integration diff, push with
`--force-with-lease` when history changed, and open the single final PR to
`main`. Without that explicit human request, leave all remote refs and PR state
unchanged.

### Required handoff

Every agent reports the task branch and worktree, base and target branches,
commit SHA(s), `git status --short`, checks run with observed results, and
remaining risks or integration steps. Disclose uncommitted work, failed checks,
conflicts, and inability to push; never imply that an uncommitted patch is a
completed isolated task.

## Security & Configuration Tips

Never commit API credentials, `.env` files, `*.local.json`, private SMPL-X parameters, checkpoints, datasets, recordings, or generated outputs. Use ignored local templates such as `stretch_mujoco/models/office_llm.example.json`, and keep credential files restricted (for example, mode `600`).
