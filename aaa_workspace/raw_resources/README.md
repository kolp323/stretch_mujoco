# NPC raw visual assets

This directory is an import staging area for local source materials that are not
committed with the repository. Runtime configuration must not refer here.

Keep NPC resources by semantic role:

```text
npc/
  accessories/<accessory_id>/<source-file>
  hair/<hair-id>/<source-file>
  textures/<material-id>/<source-file>
```

After inspection, move an approved source archive into the shared main-project
asset store at `stretch_mujoco/models/assets/humanoid/sources/`. Do not put
generated OBJ, anchors, fused frames, receipts, or videos here; those are
derived outputs.
