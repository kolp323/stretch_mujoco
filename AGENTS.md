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
- This is an NPC appearance-acceptance standard. Include close front, rear, left, right, and top
  shots; every view must exercise animation frames rather than present a single static mesh.
- Treat occlusion, colour/material stability, alpha-frame switching (no flicker), and missing
  mesh/texture/anchor assets as explicit acceptance checks. Preserve MuJoCo/asset-loader errors;
  do not hide a failed load with a fallback model.
- Deliver the MP4 with a machine-readable acceptance report that records the scene, NPC, native
  lighting policy, each shot, and any failed check. Generated videos and reports remain local
  review artifacts unless the task explicitly requests committing them.

### Required Rendering Workflow

Use `tools/render_npc_acceptance_video.py` with an office MJCF that already includes the target
NPC population and its validated assets. For a production NPC or OBJ accessory, pass the composed
office-and-NPC MJCF rather than a standalone generated NPC-only scene, so that the office's native
lighting and real furniture are rendered.

```bash
MUJOCO_GL=egl .venv/bin/python tools/render_npc_acceptance_video.py \
  --scene path/to/composed_office_npcs.xml \
  --output /tmp/npc_office_acceptance.mp4 \
  --npc-id employee_01
```

The default standing review position is the unobstructed office aisle. Use
`--standing-position X Y Z` only when the target office layout needs a different clear location.
The script emits the MP4 and a sibling `*.acceptance.json` report. Review
all five labelled shots and accept only a report with `passed: true`, no transparency or colour
failures, `animation_passed: true`, and `occlusion_passed: true` for every shot. Missing mesh,
texture, anchor, or XML
assets must remain a failing loader error; do not generate a substitute video.

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

## Shared NPC Asset Store

The primary `stretch_mujoco` worktree is the single local owner of the entire
`aaa_workspace/` tree and the entire `stretch_mujoco/models/` tree, including
scene definitions, recipes, manifests, source, restricted, and generated asset
payloads. Every other maintained worktree must expose both paths as symbolic
links to the corresponding primary-worktree directories; it must not keep a
branch-local real directory or partial copy. All content placed in either tree
is therefore shared by every branch.

Organize `aaa_workspace/` by purpose: use `docs/` for shared documentation,
`docs/workstreams/` for preserved workstream records, `task/` for plans and
checklists, `raw_resources/` for unreviewed intake, and `experiments/` for
experimental inputs and outputs grouped by subsystem and experiment. Keep
different roles isolated, and add a new top-level category only when none of
these fits. `aaa_workspace/docs/current.md` is the integrated current source of
truth; workstream records must not silently supersede it.

This `AGENTS.md` is also the single authoritative repository instruction file.
Every other maintained worktree must expose its root `AGENTS.md` as a symbolic
link to the primary worktree's file; it must not keep an independent regular-file
copy. Edit instructions only here, and verify every link after moving or
renaming a worktree.

- Keep unreviewed imports under `aaa_workspace/raw_resources/`. Runtime
  configuration must not reference this staging area.
- Move an approved local source archive to
  `stretch_mujoco/models/assets/humanoid/sources/`; keep licensed model and
  motion inputs under `private/`; write validated manifests, meshes, textures,
  anchors, and receipts under `generated/`.
- Keep rebuild contracts such as recipes, runtime configs, selection examples,
  README files, Python code, XML, and redistributable Git-tracked assets in the
  repository. They are not local asset projections.
- In another worktree, verify that both `aaa_workspace` and
  `stretch_mujoco/models` resolve to the primary worktree before changing
  shared content. `tools/link_npc_shared_assets.py` is retained only for
  compatibility with worktrees that have not yet adopted the whole-directory
  links; do not use it to replace or mutate an established shared models link.
- When development produces a reviewed resource, verify its provenance,
  license status, SHA-256/manifest contract, and relevant runtime or visual
  acceptance first. Then place it in the primary worktree's owning directory
  and recreate the worktree projection. Never promote an unverified preview by
  relabeling or copying it into a production manifest.
- Never replace the shared `aaa_workspace` link with a real directory during
  branch development, and never replace the shared `stretch_mujoco/models`
  link with a branch-local model tree. Put new branch work directly into the
  appropriate shared directory, and keep filenames or workstream
  subdirectories unambiguous when concurrent work could collide.

## Agent Git Isolation and Long-Lived Workstreams

NPC development uses maintained long-lived workstreams rather than a new branch
for every small task:

- `feat/npc/appearance-optimization` owns NPC models, materials, accessories,
  appearance configuration, and their focused tests and tools.
- `feat/npc/action-optimization` owns animation, locomotion, action execution,
  controller/runtime behavior, action-validation recording, and their focused
  tests and tools.
- `feat/npc/resource-preprocessing` owns reproducible intake and preprocessing
  work that prepares candidates for the primary shared asset store. It does not
  own a separate resource copy.

`main` remains the original-project baseline, and `feat/npc-system` remains the
NPC integration branch. Never commit or push directly to `main`; only the
designated integration owner may update `feat/npc-system`.

### Branch and worktree limits

- Do not create a branch or worktree for an individual fix, experiment, test,
  asset adjustment, or documentation update. Commit that work atomically to the
  applicable maintained workstream.
- No additional `feat/npc/*`, `fix/npc/*`, `test/npc/*`, or `docs/npc/*`
  branch may be created unless a human explicitly approves the new long-lived
  boundary before creation.
- Use only the maintained worktrees
  `stretch_mujoco-npc-action-optimization`,
  `stretch_mujoco-npc-appearance-optimization`, and
  `stretch_mujoco-npc/resource-preprocessing` for routine NPC work.
- Only one agent may write in a workstream at a time. Before editing, confirm
  the branch and inspect `git status --short` and `git diff --name-only`. Stop if
  another agent or the user has unrelated uncommitted work there.
- Treat every pre-existing modification, untracked file, ignored file, and
  commit as user-owned. Never stash, reset, clean, overwrite, move, squash, or
  discard it merely to obtain a clean worktree.
- If work crosses workstreams, assign ownership by its primary behavior and
  keep the cross-stream part minimal and explicit; do not create another branch
  merely to coordinate it.
- Configure only this repository's local commit identity as
  `kolp323 <1317416016@qq.com>`; never change the global Git identity.

### Produce atomic, reviewable commits

An agent may create local commits without another prompt while it owns the
applicable workstream. Each commit must represent one coherent change, be
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
are recorded for handoff and the PR. Never claim an unrun check passed.

After the final commit, the task worktree should be clean. Any intentional
uncommitted file must be within task scope and explicitly reported; unrelated
or unexplained residue means the task is not ready for PR.

### Remote and pull-request policy

Workstream agents must not push branches or create, update, close, or otherwise
operate on pull requests unless a human explicitly requests that exact remote
action. GitHub access being available is not authorization.

There are no routine workstream PRs. A PR may be created only after the entire
NPC module is ready for integration and a human explicitly requests it. At that
point, only the integration owner may synchronize and review from a clean
worktree:

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
or generated artifacts. If the explicitly authorized integration changes
already-pushed history, use `git push --force-with-lease`; never use an
unguarded force push. Without explicit human authorization, leave all remote
refs and PR state unchanged.

### Required handoff

Every agent reports the task branch and worktree, base and target branches,
commit SHA(s), `git status --short`, checks run with observed results, and
remaining risks or PR steps. Disclose uncommitted work, failed checks,
conflicts, and inability to push; never imply that an uncommitted patch is a
completed isolated task.

## Security & Configuration Tips

Never commit API credentials, `.env` files, `*.local.json`, private SMPL-X parameters, checkpoints, datasets, recordings, or generated outputs. Use ignored local templates such as `stretch_mujoco/models/office_llm.example.json`, and keep credential files restricted (for example, mode `600`).
