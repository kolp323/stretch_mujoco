# Local SMPL-family parameters

Place licensed model parameters in this directory. Everything except this
README is ignored by Git and must not be redistributed by the project.

Recommended layout:

```text
private/
└── smplx/
    └── SMPLX_NEUTRAL.npz
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
  --gender neutral
```

Official parameters require registration and acceptance of the model license.
The generated OBJ is only for use permitted by that license.
