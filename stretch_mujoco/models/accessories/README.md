# NPC OBJ/GLB accessory tuning

`*.recipe.json` is the only persisted pose source. Generated OBJ frames and receipts are disposable
runtime projections.

Open the interactive geometry tuner from the repository root:

```bash
.venv/bin/python tools/tune_npc_obj_accessory.py \
  --runtime-config stretch_mujoco/models/accessories/baseball_cap_v1.runtime.json
```

The blue mesh is the accessory and the tan mesh is the production body's idle reference frame.
Use the seven sliders for uniform scale, X/Y/Z translation, pitch, roll, and yaw. Switch between
front, left, back, right, and top views or drag the plot to orbit. `Save recipe` (or `Ctrl+S`)
atomically writes the current values to the configured recipe; `Reset` restores the values loaded
when the window opened.

The coordinates match the NPC's local MuJoCo frame: X is lateral, positive Y moves toward the
back of the head, and positive Z moves upward. Pitch rotates around X, roll around Y, and yaw
around Z. Use `--reference-clip walk --reference-frame 12` (or another valid frame) to inspect a
non-idle pose, and `--full-body` when a head crop is not suitable for the accessory.

The preview calls the same vertex and anchor transformations as the production fusion builder. It
does not regenerate every animation frame while dragging. After saving, rebuild the runtime scene
to apply the pose to every clip:

```bash
.venv/bin/python tools/build_npc_scene.py \
  --accessory-runtime-config stretch_mujoco/models/accessories/baseball_cap_v1.runtime.json \
  --output stretch_mujoco/models/.npc_accessory_tuned_office.xml \
  --include-base-scene
```

For a non-interactive smoke check or a remote machine without a display:

```bash
MPLBACKEND=Agg .venv/bin/python tools/tune_npc_obj_accessory.py \
  --runtime-config stretch_mujoco/models/accessories/baseball_cap_v1.runtime.json \
  --snapshot /tmp/baseball_cap_tuner.png
```
