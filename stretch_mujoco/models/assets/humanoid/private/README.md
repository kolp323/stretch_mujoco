# Local SMPL-family parameters

Place licensed model parameters in this directory. Everything except this
README is ignored by Git and must not be redistributed by the project.

Recommended layout:

```text
private/
└── smplx/
    └── SMPLX_NEUTRAL.npz
```

AMASS archives and intake outputs are local as well. Keep the original archive
or extracted source under `motion_sources/`; keep selected baker inputs under
`motions/` and their receipts under `motion_receipts/`. All are ignored by Git.
Do not commit them or copy a receipt between separately licensed downloads.

List candidate sequences without extracting an archive:

```bash
uv run prepare_amass_npc_motion list \
  --source <local-amass-archive-or-directory>
```

Prepare selected clips with a local schema-v1 selection JSON. The selection
must name the AMASS license, source ID, frame range, target FPS, and optional
reverse flag. The command refuses to overwrite a pre-existing output or receipt:
Copy `../amass_selection.example.json` into the ignored local directory and
replace every source ID and frame range only after visual review.

```bash
uv run prepare_amass_npc_motion prepare \
  --source <local-amass-archive-or-directory> \
  --selection <local-selection.json> \
  --output-dir stretch_mujoco/models/assets/humanoid/private/motions \
  --receipt-dir stretch_mujoco/models/assets/humanoid/private/motion_receipts
```

Install the CPU-only conversion runtime:

```bash
uv pip install --torch-backend cpu torch smplx
```

Generate the visual mesh and lightweight animation assets:

```bash
uv run prepare_smplx_npc \
  --model-root stretch_mujoco/models/assets/humanoid/private \
  --model-type smplx \
  --gender neutral

uv run bake_smplx_animations \
  --model-root stretch_mujoco/models/assets/humanoid/private \
  --gender neutral \
  --motion-root stretch_mujoco/models/assets/humanoid/private/motions
```

Official parameters require registration and acceptance of the model license.
The generated OBJ is only for use permitted by that license.
