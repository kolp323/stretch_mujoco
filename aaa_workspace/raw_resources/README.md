# NPC raw visual assets

This directory contains local source materials that are not committed with the
repository. Runtime configuration may refer to these files by repository-relative
path and must fail explicitly if a required source is unavailable.

Keep NPC resources by semantic role:

```text
npc/
  accessories/<accessory_id>/<source-file>
  hair/<hair-id>/<source-file>
  textures/<material-id>/<source-file>
```

The source archive for `cap_source_v1` is
`npc/accessories/cap_source_v1/cap_source_v1.zip`. Do not put generated OBJ,
anchors, fused frames, receipts, or videos here; those are derived outputs.
