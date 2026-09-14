# NPC OBJ/GLB accessory tuning

`*.recipe.json` is the only persisted pose source. Generated OBJ frames and receipts are disposable
runtime projections.

## Opaque hair-card fallback

Recipe schema v5 adds `render_policy`, which is the sole source for an OBJ
accessory's runtime-facing visual treatment. `double_sided: true` emits a
reversed-winding partner for every accessory triangle in fused frames. An
optional `surface_fallback` with `mode: "smplx_head_uv_v1"` bakes an opaque
hair-colour underlay onto selected SMPL-X head UV triangles.

`build_npc_fused_accessory.py` writes the mesh frames, target-only body atlas,
mask, receipt, fused manifest, runtime population, and (when applicable) a
derived appearance-catalog identity from that recipe. The source OBJ, source
atlas, manifest, population, and catalog are read-only inputs. Runtime scenes
must be built through `build_npc_scene.py --accessory-runtime-config`; manually
editing generated PNGs, receipts, manifests, or frames is unsupported.

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
