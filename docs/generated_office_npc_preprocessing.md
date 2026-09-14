# Generated Office NPC preprocessing

The ten authored Office scenes are kept as their original catalog entries.
Run the preprocessing tool whenever their manifests, collision geometry, or
the selected production NPC roster changes:

```bash
.venv/bin/python tools/preprocess_generated_office_npcs.py
```

The command is deterministic and idempotent.  It derives free navigation-grid
locations from each manifest zone and its compiled collision geometry; no
machine-specific source paths or hand-edited per-scene coordinates are used.

It writes these source-controlled artifacts:

- `stretch_mujoco/models/assets/office_scenes/*_npc.xml`: source scene copies
  with NPC-only sites appended.
- `stretch_mujoco/models/generated_office_npc/populations/`: three-NPC
  population derivatives that retain production appearance, animation, needs,
  personality, and dialogue settings.
- `stretch_mujoco/models/generated_office_npc/semantics/`: real body/site
  bindings, including the manifest's first graspable object and grasp site.
- `stretch_mujoco/models/generated_office_npc/trajectory_profiles/`: scene
  digest-pinned anchors and route contracts.
- `stretch_mujoco/models/generated_office_npc/receipts/`: input and generated
  SHA-256 provenance plus derived site poses.

The `*_npc.xml` scenes provide three separated spawn sites; work, meeting,
lounge and snack anchors; opposing conversation and handover stations; and
Stretch request, delivery, and rendezvous locations.  Seat approach points
are deliberately not declared: generated furniture does not expose a stable
seat semantic/body contract across all ten layouts.

For runtime composition, keep generated scene files out of source directories:

```bash
mkdir -p outputs/generated_office_npc
.venv/bin/python -m stretch_mujoco.npc.composition \
  --population stretch_mujoco/models/generated_office_npc/populations/office_01_linear_bench.population.json \
  --output outputs/generated_office_npc/office_01_linear_bench.xml
```

The composition receipt and portable wrapper are runtime output and belong in
`outputs/`, not in `models/`.

Validate all derivatives with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  .venv/bin/pytest -q tests/test_generated_office_npc_preprocessing.py
```
