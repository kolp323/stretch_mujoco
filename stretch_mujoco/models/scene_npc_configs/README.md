# Unified scene NPC configuration

`catalog.json` lists the 20 active `scene_npc_config/v1` documents. These JSON files are the
authoritative per-scene inputs for semantic registration, population, business routes, and runtime
failure policy; generated semantic/population/trajectory files are projections only.

- `office/*.json` uses manifest category rules, office zones, 8 cm navigation cells, 6 cm extra
  clearance, and the variable rosters in `office_population_plans.json`.
- `home/*.json` uses exact semantic-name taxonomy, per-home room seed overlays, 6 cm navigation
  cells, no clearance beyond the real 16 cm NPC torso radius, and migrated household spawn/activity
  targets.
- `../semantic_policies/home_scene_overrides.json` is the reviewed instance-exception extension
  point. It is currently empty; an exemption must include a concrete reason.
- `../generated_scene_npc/provenance/` records how legacy home slots and room-hint anchors were
  projected into formal per-scene resources.

Regenerate source resources, run the strict config-only audit, and publish an atomic content-addressed
active build with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/generate_scene_npc_resources.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/audit_active_scene_semantics.py \
  --report stretch_mujoco/models/generated_scene_npc/coverage/active_inventory.json \
  --compile-output-root stretch_mujoco/models/generated_scene_npc/active
```

The active runtime entry point is `../generated_scene_npc/active/active_catalog.json`.
