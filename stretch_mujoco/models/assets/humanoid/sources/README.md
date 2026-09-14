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

For example, the versioned `baseball_cap_v1` recipe resolves its source under
`npc/accessories/baseball_cap_v1/`; its source filename is a local intake
detail, not part of the public asset ID. Files here are local inputs; do not
commit them. Fused OBJ, anchors, receipts and manifests belong under the
neighbouring `generated/` directory.
