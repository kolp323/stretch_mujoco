# Shared NPC source archives

This local, untracked directory is the shared source-asset store used by NPC
runtime configurations. It is located beside generated runtime assets so a
worktree can link both directories to one main-project asset store.

Use semantic paths such as:

```text
npc/accessories/<accessory_id>/<source-file>
npc/hair/<hair-id>/<source-file>
npc/textures/<material-id>/<source-file>
```

`cap_source_v1` uses
`npc/accessories/cap_source_v1/cap_source_v1.zip`. Files here are local inputs;
do not commit them. Fused OBJ, anchors, receipts and manifests belong under the
neighbouring `generated/` directory.
