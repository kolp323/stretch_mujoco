# Generated home NPC scenes

`tools/preprocess_generated_home_npcs.py` prepares the ten HSSD home scenes for
the existing population-driven NPC runtime.  It leaves the source XML files
unchanged and writes `*_npc.xml` siblings containing three deterministic,
manifest-AABB-screened `npc_home_slot_*_site` sites, an `office_floor` navigation
alias, and an `office_overview` camera alias.  When the source HSSD manifest
contains a full-size bed, the sibling XML also receives a bed-edge
`chair_sit_site` and a farther bed-edge `seat_navigation` site; these are the
authoritative targets for the NPC sit workflow.  The sit pose faces away from
the mattress so the NPC's back remains toward the bed and no ingress through
the bed volume is required.

The corresponding population, semantic-world, and SHA-256 receipt files are
under `stretch_mujoco/models/generated_home_npc/`.  Each population contains
the same three canonical NPC identities as the office demos and assigns their
initial spawn to a different home slot.  The semantic sidecar exposes the
three sites as `binding: location`, `target: home`, with unique `slot_id`
values, so `create_mujoco_action_driver(..., world=...)` uses the existing
`LocationSlotAllocator`.  Beds are represented as `ObjectType.BED` with
`furniture_kind: bed`, allowing the same action driver to approach, align, and
sit on a declared bed instead of an unbound floor point.  Scenes without a bed
asset retain the generic activity-point fallback.

Regenerate from the checked-in home manifests with:

```bash
.venv/bin/python tools/preprocess_generated_home_npcs.py
```

To compile or render a generated scene, the HSSD converted mesh cache must be
available at `stretch_mujoco/models/assets/home_scenes/_hssd_cache/`; that
cache is intentionally ignored by Git.  Without it, artifact/schema checks
still work, but MuJoCo cannot load the scene geometry.

`load_composed_npc_runtime()` should be given the population's scene and its
semantic sidecar explicitly:

```python
load_composed_npc_runtime(
    population_path,
    output_path,
    semantic_world_path=semantic_path,
)
```
