# Shared Models Directory

This directory is the only authoritative `stretch_mujoco/models` tree for the
local worktrees. Other maintained worktrees must expose their corresponding
`stretch_mujoco/models` path as a symbolic link to this directory.

Add reviewed model resources, scene definitions, recipes, manifests, and local
generated or restricted payloads here so every branch observes the same bytes.
Do not create a branch-local model directory. Preserve provenance, licensing,
hash receipts, and runtime or visual validation requirements when promoting a
resource.

Conflicting historical branch variants retained during the shared-directory
migration are stored under
`aaa_workspace/docs/workstreams/<workstream>/models/`; they are reference
snapshots and do not supersede files in this directory.
