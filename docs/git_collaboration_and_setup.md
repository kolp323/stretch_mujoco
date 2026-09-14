# Git collaboration and repository hygiene

`AGENTS.md` is the authoritative policy for the maintained NPC worktrees. This page summarizes
the developer workflow without redefining branch ownership.

## Before editing

Work from the repository root and inspect existing work before making broad changes:

```bash
git status --short --branch
git diff --name-only
git submodule status
git worktree list
```

Treat every existing modification, untracked file, ignored file, and local commit as user-owned.
Do not use `git clean`, `git reset --hard`, or blanket staging to obtain a clean tree. The primary
worktree owns the real `aaa_workspace/` and `stretch_mujoco/models/` directories; maintained NPC
worktrees must link to them as described in `AGENTS.md`.

## What belongs in Git

Track source code, tests, documentation, dependency metadata, schemas, recipes, small example
configuration, and redistributable runtime assets. Do not track:

- `.env`, `*.local.json`, API credentials, or machine-specific paths;
- datasets, checkpoints, private SMPL-X inputs, source archives, or licensed raw assets;
- `outputs/`, recordings, logs, caches, temporary XML, or generated diagnostic images;
- virtual environments, build products, or editor state.

Use `git check-ignore -v <path>` to verify the applicable rule. Ignoring a file does not delete it
and does not untrack a file that is already committed. If a tracked file should become local-only,
review that change explicitly; do not remove data merely to make `git status` shorter.

The repository intentionally tracks several large robot and office OBJ meshes required at
runtime. Treat those as versioned assets, not accidental checkpoints. New large assets need a
license/provenance review and should use an external artifact store when they are not required to
run the checked-in examples.

## Environment and dependencies

Use the checked-in lockfile:

```bash
uv python install 3.10
uv sync --extra dev
uv lock --check
```

The current `openpi-client` source override expects
`../openpi/packages/openpi-client`. Provision that sibling checkout before resolving or updating
the lockfile. Optional external datasets are configured through the CLI/environment variables in
the root README; their absolute locations must not be committed.

## Validation and commits

Run the narrowest relevant checks while editing, then the root suite when practical:

```bash
uv run pytest -q tests/test_office_scene.py
MUJOCO_GL=egl uv run pytest -q
uv run pre-commit run --all-files
git diff --check
```

Before committing, inspect and stage explicit paths only:

```bash
git status --short
git diff -- path/to/file
git add path/to/file
git diff --cached --check
git diff --cached --name-status
git diff --cached
git commit -m "type(scope): concise summary"
```

Never use `git add .`, `git add -A`, or `git commit -a` in a worktree containing unrelated work.
Do not push, force-push, rewrite history, or operate on a pull request without explicit authority.
The exact maintained branch/worktree rules and required handoff fields remain in `AGENTS.md`.

## Troubleshooting

- Tests under `third_party/` are being collected: run from the root using its `pyproject.toml`;
  the configured discovery root is `tests/`.
- Generated files appear beside source: redirect runtime outputs with
  `STRETCH_MUJOCO_OUTPUT_DIR`; disposable conversions use `STRETCH_MUJOCO_CACHE_DIR`.
- An external dataset path appears in a diff: replace it with a repository-relative runtime path,
  a dataset-relative provenance locator, or a documented environment variable.
- A submodule is empty: run `git submodule update --init`, then verify `git submodule status`.
