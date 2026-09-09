# Shared Workspace

`aaa_workspace/` is a single shared workspace owned by the primary
`stretch_mujoco` worktree. Every other maintained worktree exposes this whole
directory through a symbolic link; branch-local copies are not allowed.

Keep content separated by purpose:

- `docs/`: shared implementation records and resource documentation.
- `docs/workstreams/`: preserved records from individual development streams;
  `docs/current.md` remains the integrated current source of truth.
- `task/`: local plans, checklists, and task notes.
- `raw_resources/`: unreviewed source archives and intake material. Runtime
  configuration must never reference this directory.
- `experiments/`: reproducible experimental inputs and outputs, grouped first
  by subsystem and then by experiment name.

Create a new top-level subdirectory only when none of these roles fits. Do not
mix raw inputs, experimental outputs, documentation, and task notes in one
directory. A reviewed resource must be validated and promoted to its owning
repository asset directory in the primary worktree before production use.
