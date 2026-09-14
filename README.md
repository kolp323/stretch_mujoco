# Stretch Mujoco

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/stretch_mujoco.png" title="Stretch In Kitchen" width="100%">

This library provides a simulation stack for Stretch, built on [MuJoCo](https://github.com/google-deepmind/mujoco). There is position control for the arm, head, and gripper joints, velocity control for mobile base, calibrated camera RGB + depth imagery, 2D spinning lidar scans, and more. There is a visualizer that supports [user interaction](https://youtu.be/2P-Dt-Jfd6U), or a more efficient headless mode. There is a [ROS2 package](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_simulation), built on this library, that works with Nav2, Web Teleop, and more. There is 100s of permutations of Robocasa-provided kitchen environments that Stretch can spawn into. The MuJoCo API can be used for features like deformables, procedural model generation, SDF collisions, cloth simulation, and more.

Check out the [highlight reel](https://www.youtube.com/watch?v=SWPJt67IB0Q) for features that have been recently added.

## Project layout

The repository has one primary Python package and one optional agent package:

| Path | Purpose |
| --- | --- |
| `stretch_mujoco/` | Simulator client/server, robot interfaces, navigation, NPC runtime, recording, and packaged MuJoCo assets |
| `examples/` | Runnable simulation, teleoperation, data-collection, policy-evaluation, and office demos |
| `tools/` | Asset conversion, scene generation, validation, and rendering utilities |
| `tests/` | Root pytest suite; vendor tests under `third_party/` are deliberately excluded |
| `strech_codex/` | Optional MCP/Codex agent package with its own `pyproject.toml` and tests |
| `docs/` | User and contributor documentation |
| `third_party/` | Robocasa and Robosuite Git submodules |
| `outputs/` | Ignored runtime logs, recordings, evaluations, and generated reports |
| `aaa_workspace/` | Shared local NPC intake, experiments, task notes, and implementation records |

Domain configuration stays with its owner. Robot/scene assets and NPC manifests live under
`stretch_mujoco/models/`; trajectory profiles live under `stretch_mujoco/npc/trajectory_profiles/`.
Machine-specific paths and credentials do not belong in those files.

## Installation

Python 3.10 or newer is supported; use Python 3.10 when installing the Robocasa extra. From a
clone with submodules:

```bash
git submodule update --init
uv python install 3.10
uv sync --extra dev
uv run python -c "import mujoco, stretch_mujoco; print(mujoco.__version__)"
```

The current lockfile resolves `openpi-client` from the sibling path
`../openpi/packages/openpi-client`. Provision that checkout before refreshing the lockfile. The
base simulator does not need an OpenPI server at runtime; OpenPI evaluation commands do.

## Runtime configuration

CLI options take precedence. The following optional environment variables remove
machine-specific paths from code and generated assets:

| Variable | Meaning | Default |
| --- | --- | --- |
| `STRETCH_MUJOCO_OUTPUT_DIR` | Logs, recordings, evaluations, and reports | `<repo>/outputs` |
| `STRETCH_MUJOCO_CACHE_DIR` | Disposable converted assets and caches | system temp directory |
| `STRETCH_MUJOCO_HSSD_ROOT` | HSSD dataset root | required by HSSD tools |
| `STRETCH_MUJOCO_ROBOTWIN_ROOT` | RoboTwin `assets/objects` root | required by conversion tool |
| `STRETCH_MUJOCO_OPENPI_ROOT` | OpenPI checkout root | required by checkpoint evaluator |
| `STRETCH_MUJOCO_GRASPGEN_HOST` | GraspGen service host | `127.0.0.1` |

`.env.example` documents these names, but the project does not implicitly load `.env`; export
variables in the shell or pass the corresponding CLI option. LLM credentials belong only in the
ignored `stretch_mujoco/models/office_llm.local.json`, copied from
`office_llm.example.json` and restricted to the current user.

## Data and model preparation

The base robot scenes and redistributable assets are included. Optional workflows require their
licensed upstream data:

```bash
uv run prepare_smplx_npc --help
uv run prepare_amass_npc_motion --help
uv run build_npc_scene --help
uv run compose_npc_scene --help
uv run python tools/convert_robotwin_objects.py --help
uv run python tools/build_office_asset_catalog.py --help
```

Keep downloaded datasets, checkpoints, private SMPL-X inputs, and generated recordings out of
Git. Store raw NPC intake under `aaa_workspace/raw_resources/`, private humanoid inputs under
`stretch_mujoco/models/assets/humanoid/private/`, and reproducible runtime artifacts under the
configured output root.


## Getting Started
Start with Google Colab:

 - Getting Started Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hello-robot/stretch_mujoco/blob/main/docs/getting_started.ipynb)

**or** follow these instructions on your computer:

First, install [`uv`](https://docs.astral.sh/uv/#getting-started). Uv is a package manager that we'll use to run this project.

Then, clone this repo:

```
git clone https://github.com/hello-robot/stretch_mujoco --recurse-submodules
cd stretch_mujoco
```

> If you've already cloned the repo without `--recurse-submodules`, run `git submodule update --init` to pull the submodule.

Lastly, run the simulation:

```
uv run launch_sim
```

> Note: If you see a build error mentioning `evdev` on Linux, install your distribution's
> Python development headers (for example, `sudo apt install python3-dev` on Debian/Ubuntu).

To exit, press `Ctrl+C` in the terminal.

<p>
    <img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/camera_streams.png" title="Camera Streams" height="250px">
    <img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/stretch3_in_mujoco.png" title="Camera Streams" height="250px">
</p>

> On MacOS, if `mjpython` fails to locate `libpython3.10.dylib` and `libz.1.dylib`, run these commands:
```shell
# Before proceeding, please reload your terminal and/or IDE window, to make sure the correct UV environment variables are loaded.

source .venv/bin/activate

# When `libpython3.10.dylib` is missing, run:
PYTHON_LIB_DIR=$(python3 -c 'from distutils.sysconfig import get_config_var; print(get_config_var("LIBDIR"))')
ln -s "$PYTHON_LIB_DIR/libpython3.10.dylib" ./.venv/lib/libpython3.10.dylib

# When `libz.1.dylib` is missing, run:
export DYLD_LIBRARY_PATH=/usr/lib:$DYLD_LIBRARY_PATH
```

## Example Scripts

[Office and snack scene](./examples/office_scene.py)

Launch Stretch 3 in a furnished office with work, meeting, lounge, and break areas.
The four graspable snacks use imported textured 3D meshes with lightweight collision proxies:

```
uv run examples/office_scene.py
```

Use `--headless` to run the same scene without the interactive viewer.

Cycle through the pre-baked SMPL-X NPC animations:

```
uv run examples/office_scene.py --animation-demo
```

The `sit` request makes the NPC walk around the workstation, approach the chair,
turn, and lower into the seat. It uses the right office chair by default. Select
the other chair with `--npc-chair chair_left`:

```
uv run examples/office_scene.py --npc-animation sit --npc-chair chair_left
```

The office semantic world maps MuJoCo entities to typed objects, relations, and
agent-facing interaction points. Inspect the graph without launching the viewer:

```
uv run examples/office_semantics.py --object-id document_report
```

When the office simulator is running, `sim.semantic_world` exposes relation queries
and updates, while `sim.pull_semantic_state()` returns the latest 5 Hz world-space
poses for semantic objects and interaction points. See
[Office Semantic World](./docs/office_semantics.md) for relation direction and
Agent update rules.

Run the deterministic employee action protocol and create a robot delivery task:

```
uv run examples/office_agents.py --complete-task
```

The runtime contains no LLM calls. `ActionCommand` accepts only the built-in action
enum, and `OfficeAgentRuntime` owns validation, reservations, conflicts, needs,
schedules, memory, execution state, robot requests, and result verification. See
[Office Employee Agents](./docs/office_agents.md) for the simulator loop and
physical-controller handoff contract.

For NPC population configuration, versioned asset manifests, appearance
recipes, scene generation, command receipts, and conversation integration, see
[NPC System Guide](./docs/npc_system.md).

Inspect the seeded low-compute state machine, utility decisions, daily events, and
event-only LLM triggers:

```
uv run examples/office_autonomy.py --seconds 60
```

See [Low-Compute Autonomous Behavior](./docs/low_compute_behavior.md) for update
frequencies, utility factors, anti-repetition behavior, and the LLM call boundary.
Local provider credentials belong in the ignored
`stretch_mujoco/models/office_llm.local.json`; start from the adjacent
`office_llm.example.json` template and set its permissions to `600`.

[Keyboard teleop](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/keyboard_teleop.py)

```
uv run examples/keyboard_teleop.py
```

[Gamepad teleop](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/gamepad_teleop.py)

Control Stretch in simulation using any xbox type gamepad (uses xinput)

```
uv run examples/gamepad_teleop.py
```

[Robocasa environments](https://github.com/hello-robot/stretch_mujoco/tree/main/examples/robocasa_environment.py)

```
# Setup
uv pip install -e ".[robocasa]"
uv pip install -e "robocasa@third_party/robocasa"
uv pip install -e "robosuite@third_party/robosuite"
uv run third_party/robosuite/robosuite/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/download_kitchen_assets.py

# Run sim
uv run examples/robocasa_environment.py
```

Ignore any warnings.

<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/robocasa_scene_1.png" title="Camera Streams" width="300px">
<img src="https://github.com/hello-robot/stretch_mujoco/raw/main/docs/images/robocasa_scene_camera_data.png" title="Camera Streams" width="300px">

## Training, inference, and evaluation

This repository does not contain a general-purpose model training loop. It provides simulation,
dataset collection, deterministic replay, and evaluation clients. Training OpenPI or another
policy happens in that model's repository; use the tools here to produce or evaluate compatible
episodes:

```bash
# Collect local episodes (written below datasets/, which is ignored).
uv run examples/collect_random_grasp_episodes.py --help
uv run examples/collect_random_object_grasp_episodes.py --help

# Evaluate a running OpenPI policy server.
uv run examples/evaluate_openpi_policy.py --help

# Evaluate retained checkpoints by starting the OpenPI policy server per checkpoint.
uv run python tools/evaluate_openpi_checkpoints.py --help
```

NPC and office workflows are simulation/runtime features rather than learned-policy training.
Use `uv run examples/office_scene.py --headless` for a minimal scene compile and the commands in
the NPC System Guide for manifest-backed NPC generation and validation.

## Outputs and reproducibility

Use one subdirectory per run below `outputs/`, for example
`outputs/<workflow>/<YYYYMMDD_HHMMSS>_<label>/`. Keep the effective input configuration, random
seed, summary JSON, and logs together; put large videos/checkpoints below the same ignored run
directory or external storage. Tools that create source-owned model assets still write to their
documented `stretch_mujoco/models/` locations.

For a reproducible handoff, record the Git commit, `uv.lock`, command line, relevant environment
variables, dataset/asset checksums, and random seed. Do not commit `outputs/`, `datasets/`, caches,
local credential files, or private model inputs.

## Validation

Run checks from the repository root:

```bash
uv lock --check
uv run pytest -q
uv run pytest -q tests/test_office_scene.py
uv run pre-commit run --all-files
MUJOCO_GL=egl uv run examples/office_scene.py --headless
```

The root pytest configuration only discovers `tests/`; run the optional agent package separately
with `uv run --project strech_codex --extra dev pytest`. Rendering, external services, private assets, and GPU
workflows may need additional local setup and should report their unmet prerequisite explicitly.

## Common problems

- `openpi-client` cannot be resolved: provision `../openpi/packages/openpi-client` before running
  `uv lock` or `uv sync` with the current source override.
- HSSD/RoboTwin/OpenPI data cannot be found: pass the tool's root option or export the matching
  `STRETCH_MUJOCO_*_ROOT` variable.
- GLFW/OpenGL initialization fails on a headless host: set `MUJOCO_GL=egl` and use `--headless`.
- An NPC asset is missing: initialize the approved private/generated asset projection; do not
  replace it silently with a preview asset.
- A run created many local files: keep them under `outputs/` or the documented ignored workspace
  area, then inspect with `git status --short --ignored` before staging.

## Writing Code

Use the [`StretchMujocoSimulator`](./stretch_mujoco/stretch_mujoco_simulator.py) class to:

 * start the simulation
 * position control the robot's ranged joints
 * velocity control the robot's mobile base
 * read joint states
 * read camera imagery

Try the code below using `uv run ipython`. For advanced Mujoco users, the class also exposes the `mjModel` and `mjData`. See the [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html).

```python
from stretch_mujoco import StretchMujocoSimulator

if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start(headless=False) # This will open a Mujoco-Viewer window
    
    # Poses
    sim.stow()
    sim.home()
    
    # Position Control 
    sim.move_to('lift', 1.0)
    sim.move_by('head_pan', -1.1)
    sim.move_by('base_translate', 0.1)

    sim.wait_until_at_setpoint('lift')
    sim.wait_while_is_moving('base_translate')
    
    # Base Velocity control
    sim.set_base_velocity(0.3, -0.1)
    
    # Get Joint Status
    from pprint import pprint
    pprint(sim.pull_status())
    """
    Output:
    {'time': 6.421999999999515,
     'base': {'x_vel': -3.293721562016785e-07,'theta_vel': -3.061556698064456e-05},
     'lift': {'pos': 0.5889703729548038, 'vel': 1.3548342274419937e-08},
     'arm': {'pos': 0.09806380391427844, 'vel': -0.0001650879063921366},
     'head_pan': {'pos': -4.968686850480367e-06, 'vel': 3.987855066304579e-08},
     'head_tilt': {'pos': -0.00451929555883404, 'vel': -2.2404905787897265e-09},
     'wrist_yaw': {'pos': 0.004738908190630005, 'vel': -5.8446467640096307e-05},
     'wrist_pitch': {'pos': -0.0033446975569971366,'vel': -4.3182498418896415e-06},
     'wrist_roll': {'pos': 0.0049449466225058416, 'vel': 1.27366845279872e-08},
     'gripper': {'pos': -0.00044654737698173895, 'vel': -8.808287459130369e-07}}
    """
    
    # Get Camera Frames
    camera_data = sim.pull_camera_data()
    pprint(camera_data)
    """
    Output:
    {'time': 80.89999999999286,
     'cam_d405_rgb': array([[...]]),
     'cam_d405_depth': array([[...]]),
     'cam_d435i_rgb': array([[...]]),
     'cam_d435i_depth': array([[...]]),
     'cam_nav_rgb': array([[...]]),
     'cam_d405_K': array([[...]]),
     'cam_d435i_K': array([[...]])}
    """
    
    # Kills simulation process
    sim.stop()
```

Note that the `if __name__ == "__main__":` guard is necessary, as explained in the [Python Docs](https://docs.python.org/3/library/multiprocessing.html#:~:text=For%20an%20explanation%20of%20why%20the%20if%20__name__%20%3D%3D%20%27__main__%27%20part%20is%20necessary%2C%20see%20Programming%20guidelines.).

### Loading Robocasa Kitchen Scenes

The `stretch_mujoco.robocasa_gen.model_generation_wizard()` method gives you:

- Wizard/API to generate a kitchen model for a given task, layout, and style.
- If layout and style are not provided, it will take you through a wizard to choose them in the terminal.
- If robot_spawn_pose is not provided, it will spawn the robot to the default pose from robocasa fixtures.
- You can also write the generated xml model with absolutepaths to a file.

```python
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard

# Use the wizard:
model, xml, objects_info = model_generation_wizard()

# Or, launch a specific task/layout/style
model, xml = model_generation_wizard(
    task=<task_name>,
    layout=<layout_id>,
    style=<style_id>,
    write_to_file=<filename>,
)

sim = StretchMujocoSimulator(model=model)
sim.start()
```

### ROS2

You can use this simulation in ROS2 using the [`stretch_simulation` package](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_simulation) in `stretch_ros2`.

### Docs

Check out the following documentation resources:

- [Using the Mujoco Simulator with Stretch](./docs/using_mujoco_simulator_with_stretch.md)
- [Getting Started jupyter notebook](./docs/getting_started.ipynb)
- [Releasing to PyPi](./docs/releasing_to_pypi.md)
- [Contributing to this project](./docs/contributing.md)
- [Changelog](./CHANGELOG.md)

### Feature Requests and Bug reporting

All enhancements/missing features/bugfixes are tracked by [Issues](https://github.com/hello-robot/stretch_mujoco/issues) filed. Please feel free to file an issue if you would like to report bugs or request a feature addition. Pull requests are welcome! Please see the [contributing guide](./docs/contributing.md).

## Acknowledgment

The assets in this repository contain significant contributions and efforts from [Kevin Zakka](https://github.com/kevinzakka) and [Google Deepmind](https://github.com/google-deepmind), along with others in Hello Robot Inc. who helped us in modeling Stretch in Mujoco. Thank you for your contributions.
