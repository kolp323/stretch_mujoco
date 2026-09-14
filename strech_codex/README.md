# strech-codex

Codex-style agent and MCP tools for controlling MuJoCo robots through natural language.

`strech_codex` wraps the `stretch_mujoco` robot abstraction layer and navigation
algorithms as an MCP (Model Context Protocol) tool server, then provides an agent
loop that translates natural-language commands into tool-call sequences — both
offline (deterministic parser) and live (Codex / LLM-driven).

## Quick start

```bash
# From the stretch_mujoco repository root with the project virtual environment.
.venv/bin/python -m pip install -e './strech_codex[mcp,codex]'

# Offline — deterministic tool execution (no API key needed).
.venv/bin/python -m strech_codex.cli --mode offline "启动 stretch3 机器人并查询状态"

# Launch the MCP stdio server (requires mcp SDK).
.venv/bin/python -m strech_codex.mcp_server

# Live — LLM-driven orchestration via Codex SDK (requires API key).
.venv/bin/python -m strech_codex.cli --mode live --verbose "去厨房帮我拿一个苹果"
```

## Architecture

```text
                         ┌──────────────────────────┐
                         │  Natural language task    │
                         │  (NPC / user / external)  │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │     CLI  (cli.py)         │
                         │  --mode offline|live      │
                         └──────┬──────────┬─────────┘
                                │          │
                   ┌────────────▼──┐  ┌───▼──────────────┐
                   │  agent.py     │  │  codex_adapter.py │
                   │  (deterministic│  │  (Codex SDK /     │
                   │   NL parser)   │  │   LLM orchestration)
                   └──────┬────────┘  └───┬──────────────┘
                          │               │
                          └───────┬───────┘
                                  │ dispatch_tool(name, args)
                         ┌────────▼──────────────────────┐
                         │      mcp_server.py             │
                         │  tool_manifest() → 31 tools    │
                         │  build_mcp_server() → FastMCP  │
                         └──┬──────────┬──────────┬──────┘
                            │          │          │
               ┌────────────▼──┐ ┌─────▼──────┐ ┌▼──────────┐
               │ tools/        │ │ tools/     │ │ tools/     │
               │ navigation.py │ │ robot.py   │ │ scene.py   │
               │ (A*,FMM,FBE,  │ │ (3 robot   │ │ (object     │
               │  VLFM)        │ │  types)    │ │  queries)   │
               └───────┬───────┘ └─────┬──────┘ └─────┬──────┘
                       │               │              │
               ┌───────▼───────────────▼──────────────▼──────┐
               │           world/state.py                    │
               │  Shared singleton: sim, nav, explorer,       │
               │  scene_path, robot_type                     │
               └──────────────────┬──────────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────────┐
               │          stretch_mujoco                      │
               │  ┌─────────────┐  ┌───────────────────────┐  │
               │  │ robots/     │  │ navigations/           │  │
               │  │ Stretch 3   │  │ A*, FMM, FBE, VLFM    │  │
               │  │ Google Robot│  │ OccupancyGrid          │  │
               │  │ TidyBot     │  │ NavigationController   │  │
               │  └─────────────┘  └───────────────────────┘  │
               └─────────────────────────────────────────────┘
```

### Key design decisions

- **Shared world state** (`world/state.py`) — a module-level singleton holds the
  active `RobotSimulator`, `NavigationController`, and explorer instance.  All MCP
  tool functions read from and write to this singleton, so tools composed by an
  agent operate on the same MuJoCo process and occupancy grid.
- **Tool → MCP dispatch** — every tool is a plain Python function returning
  `{"success": bool, "message": str, ...}`.  The MCP server registers them
  on a `FastMCP` instance, and the agent calls them directly through
  `dispatch_tool(name, arguments)`.
- **Three tool groups** — navigation (A\*, FMM, FBE, VLFM), robot control
  (Stretch 3 / Google Robot / TidyBot), and scene queries.  Each group is a
  separate module under `tools/`.
- **Offline + live dual mode** — the deterministic parser in `agent.py` lets
  you test tool orchestration without an LLM; `codex_adapter.py` provides the
  Codex SDK path for real LLM-driven execution.

## Installation

```bash
# Core (offline agent, MCP server — no LLM dependencies).
.venv/bin/python -m pip install -e './strech_codex'

# With MCP SDK (needed for the stdio server and live mode).
.venv/bin/python -m pip install -e './strech_codex[mcp]'

# With Codex SDK (needed for live mode).
.venv/bin/python -m pip install -e './strech_codex[codex]'

# With VLM backends (needed for VLFM exploration).
.venv/bin/python -m pip install -e './strech_codex[vlfm]'

# Development.
.venv/bin/python -m pip install -e './strech_codex[dev]'
```

## Directory structure

```text
strech_codex/
├── pyproject.toml
├── README.md
├── src/strech_codex/
│   ├── __init__.py
│   ├── agent.py             # Offline NL → tool-call planner
│   ├── cli.py               # CLI entry point
│   ├── codex_adapter.py     # Codex SDK live-mode integration
│   ├── mcp_server.py        # FastMCP server + dispatch layer
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── navigation.py    # A*, FMM, FBE, VLFM tools
│   │   ├── robot.py         # Robot control tools
│   │   └── scene.py         # Scene query tools
│   └── world/
│       ├── __init__.py
│       └── state.py         # Shared world state singleton
└── tests/
    ├── test_navigation_tools.py   # Grid building, A*, FMM
    ├── test_robot_tools.py        # Robot lifecycle, status, scene fallback
    ├── test_agent_cli.py          # CLI config, planner regressions
    └── test_codex_adapter.py      # Live-mode MCP server registration
```

## World state (`world/state.py`)

A module-level `_WorldState` singleton holds all mutable state shared across
tool calls.  This pattern mirrors `robot_project/src/robot_project/robot_tools.py`.

```
                    ┌─────────────────────────┐
                    │       _WorldState        │
                    ├─────────────────────────┤
                    │ _sim: RobotSimulator     │ ← robot_init / robot_stop
                    │ _nav: NavigationController│ ← nav_build_grid
                    │ _explorer: FBEPlanner    │ ← nav_fbe_init / nav_vlfm_init
                    │       | VLFMPlanner      │
                    │ _scene_path: str | None  │ ← CLI --scene / robot_init
                    │ _robot_type: str | None  │ ← CLI --robot / robot_init
                    └─────────────────────────┘
```

Public accessors (used by tool functions):

| Function | Purpose |
|---|---|
| `get_sim()` / `set_sim()` | Active `RobotSimulator` |
| `has_sim()` | Guard before robot tools |
| `get_nav()` / `set_nav()` | Active `NavigationController` |
| `has_nav()` | Guard before navigation tools |
| `get_explorer()` / `set_explorer()` | Active `FBEPlanner` or `VLFMPlanner` |
| `has_explorer()` | Guard before exploration tools |
| `get_scene_path()` / `set_scene_path()` | Current MuJoCo scene XML path |
| `get_robot_type()` / `set_robot_type()` | `"stretch3"`, `"google_robot"`, or `"tidybot"` |
| `reset_world()` | Stop simulator, clear all state |

## MCP tools (31 total)

### Navigation tools (12)

#### Grid management

| Tool | Parameters | Description |
|---|---|---|
| `nav_build_grid` | `scene_xml?`, `bounds?`, `resolution=0.08`, `agent_radius=0.25`, `floor_geom_name="office_floor"`, `minimum_obstacle_height=0.08`, `maximum_obstacle_height=1.80`, `require_collision=True`, `exclude_prefixes?` | Rasterise MuJoCo collision geometry into a 2-D occupancy grid. Must be called before any path planner. Returns grid dimensions, free-cell count, and bounds. |
| `nav_get_grid_info` | _(none)_ | Return the current grid's dimensions, resolution, bounds, and free ratio. |
| `nav_is_free` | `x`, `y` | Check whether a world point is navigable on the current grid. |

#### A\* (A-star)

| Tool | Parameters | Description |
|---|---|---|
| `nav_astar` | `start_x`, `start_y`, `goal_x`, `goal_y`, `smoothing=True` | Plan a collision-free path using A\* on an 8-connected grid. Returns ordered waypoints, path distance, and waypoint count. |

**Algorithm details:** Standard A\* search with Euclidean distance heuristic,
8-connected neighbourhood (`√2` diagonal cost), corner-cut prevention,
and greedy path smoothing that removes intermediate waypoints when the
straight line is collision-free.

#### FMM (Fast Marching Method)

| Tool | Parameters | Description |
|---|---|---|
| `nav_fmm` | `start_x`, `start_y`, `goal_x`, `goal_y`, `smoothing=True` | Solve the Eikonal equation `|∇T| = 1` outward from the start, then extract a path via gradient descent from the goal. |

**Algorithm details:** First-order upwind FMM discretisation on a uniform grid.
The arrival-time field is computed outward from start until the goal cell is
accepted.  Gradient descent follows the steepest time decrease with
line-of-sight verification at each step.

#### Path execution

| Tool | Parameters | Description |
|---|---|---|
| `nav_execute_path` | `waypoints`, `timeout=120`, `position_tolerance=0.12`, `start_tolerance=0.5`, `max_linear_speed=0.4`, `max_angular_speed=0.8`, `control_hz=20` | Follow world-frame waypoints with closed-loop base control. Uses differential drive for Stretch 3 and omnidirectional control for Google Robot/TidyBot; always commands zero velocity before returning. |

#### FBE (Frontier-Based Exploration)

| Tool | Parameters | Description |
|---|---|---|
| `nav_fbe_init` | `scene_xml?`, `bounds?`, `resolution=0.08`, `agent_radius=0.25`, `num_rays=180`, `max_range_m=5.0`, `fov_degrees=270.0`, `min_cluster_size=3`, `explore_threshold=0.85`, `start_x=0.0`, `start_y=0.0`, `start_yaw=0.0` | Build a god grid and create an `FBEPlanner` with simulated laser scanner. Call `nav_fbe_step` in a loop to explore. |
| `nav_fbe_step` | `robot_x`, `robot_y`, `robot_yaw` | Advance exploration by one step (~10-20 Hz). Casts rays, updates local map, detects frontiers, clusters them, plans A\* path to the best frontier. Returns state, explored ratio, frontier count, and next movement command. |
| `nav_fbe_path` | _(none)_ | Return the current FBE path waypoints. |

**State machine:** `SCANNING → PLANNING → MOVING → ROTATING → FINISHED`.
Terminates when the explored ratio reaches the threshold and no reachable
frontiers remain, or after a configurable number of stagnant scans.

#### VLFM (Vision-Language Frontier Maps)

| Tool | Parameters | Description |
|---|---|---|
| `nav_vlfm_init` | `instruction`, `scene_xml?`, `vlm_type="clip"`, `device="cpu"`, `resolution=0.08`, `agent_radius=0.25`, `max_range_m=5.0`, `explore_threshold=0.85`, `allow_geometric_fallback=True`, `start_x/y/yaw` | Initialise VLFM exploration with a natural-language instruction (e.g. "Seems like there is a chair ahead."). Builds god grid, loads VLM model (CLIP / SigLIP / OpenAI / BLIP2). |
| `nav_vlfm_step` | `robot_x`, `robot_y`, `robot_yaw` | Advance VLFM by one step. Scores frontiers via the accumulated value map, enforces acyclic selection to prevent oscillation, then plans A\* to the best frontier. |
| `nav_vlfm_inject_observation` | `rgb_image_path`, `depth_image_path`, `camera_x/y/yaw`, `hfov_rad=1.2`, `min_depth=0.1`, `max_depth=5.0` | Score an RGB image against the VLFM instruction, then project the score through the camera FOV onto the spatial value map using the depth image. |

**Algorithm details:** Aligned with the upstream
[rai-opensource/vlfm](https://github.com/rai-opensource/vlfm) ITM (Image-Text
Matching) policy pattern (ICRA 2024).  At each decision step the VLM scores the
current egocentric view; the score is projected through the depth FOV onto a
spatial value map.  Frontiers are ranked by accumulated value-map scores, and an
`AcyclicEnforcer` prevents revisiting recently explored areas.  Two-channel mode
(target + exploration) switches between "seek" and "explore" based on a
configurable threshold.

### Robot control tools (16)

#### Lifecycle

| Tool | Parameters | Description |
|---|---|---|
| `robot_init` | `robot_type`, `scene_xml?`, `headless=True`, `cameras_to_use?`, `camera_hz=30.0`, `start_translation?`, `start_rotation_quat?` | Create and start a robot simulator. `robot_type` is `"stretch3"`, `"google_robot"`, or `"tidybot"`. `cameras_to_use` accepts camera enum names or the string aliases `"all_rgb"`, `"all_depth"`, `"all"`. |
| `robot_stop` | _(none)_ | Stop the simulator and clear robot state from the shared world. |

#### Joint control

| Tool | Parameters | Description |
|---|---|---|
| `robot_move_to` | `actuator_name`, `position` | Command an absolute joint position target. Resolves the actuator by enum name, enum value, or MJCF joint name. |
| `robot_move_by` | `actuator_name`, `delta` | Command a relative position increment. |
| `robot_set_base_velocity` | `v_linear=0.0`, `omega=0.0`, `v_lateral=0.0` | Set mobile base velocity. `v_lateral` only works on omnidirectional bases (Google Robot, TidyBot); Stretch 3 raises `ValueError`. |
| `robot_home` | _(none)_ | Move to the home keyframe (Stretch 3, TidyBot; Google Robot raises `NotImplementedError`). |
| `robot_stow` | _(none)_ | Move to the stow/retract keyframe. |
| `robot_wait_until_at_setpoint` | `actuator_name`, `timeout=5.0`, `position_tolerance=0.05` | Block until the actuator reaches its commanded `move_to` target. |

#### Status queries

| Tool | Parameters | Description |
|---|---|---|
| `robot_get_status` | _(none)_ | Pull the full joint-state snapshot as a dict. Fields vary by robot type. |
| `robot_get_base_pose` | _(none)_ | Return `(x, y, theta)` in world coordinates. |
| `robot_get_ee_pose` | _(none)_ | Return the 4×4 end-effector pose matrix. |
| `robot_get_camera_data` | `camera_name=""` | When `camera_name` is provided, returns that camera's image metadata (shape, dtype). Otherwise lists available RGB and depth cameras. |

#### Grasping

| Tool | Parameters | Description |
|---|---|---|
| `robot_attach_object` | `object_id` | Fix a named body to the gripper for stable grasp simulation. |
| `robot_release_object` | _(none)_ | Release the currently attached object. |

#### Introspection

| Tool | Parameters | Description |
|---|---|---|
| `robot_list_actuators` | _(none)_ | List all actuators with name, value, type (`POSITION`/`VELOCITY`/`GENERAL`), and whether they are base actuators. |
| `robot_list_cameras` | _(none)_ | List all cameras with name, MJCF name, RGB/depth classification. |

### Scene tools (3)

| Tool | Parameters | Description |
|---|---|---|
| `scene_list_objects` | _(none)_ | Enumerate free-joint bodies in the scene (movable objects). Falls back to the scene XML if the simulator backend doesn't expose `mjmodel` directly. |
| `scene_get_object_pose` | `object_name` | Return the 4×4 world pose of a named body. |
| `scene_add_world_frame` | `x=0.0`, `y=0.0`, `z=0.0` | Add a coordinate-frame marker at the given world position (viewer visualisation). |

## Agent system (`agent.py`)

### Offline planner

`plan_offline(task)` parses Chinese and English natural-language commands into
`ToolCall` sequences using regex patterns.  It recognises:

| Pattern | Generated tool calls |
|---|---|
| "构建地图" / "build grid" | `nav_build_grid` |
| "从 (sx, sy) 到 (gx, gy)" / "导航到 (x, y)" | `nav_astar` (with auto-injected `nav_build_grid` if no grid exists) |
| "启动 stretch3 / google_robot / tidybot" | `robot_init` |
| "移动 lift 到 0.5" / "move arm to 1.0" | `robot_move_to` |
| "前进" / "后退" / "左转" / "右转" | `robot_set_base_velocity` |
| "抬起手臂" / "放下手臂" | `robot_move_to(lift, …)` |
| "抓取/拿 object" / "释放" | `robot_attach_object` / `robot_release_object` |
| "状态" / "位置" / "在哪" | `robot_get_status` + `robot_get_base_pose` |
| "探索" / "扫描" | `nav_fbe_init` (or `nav_vlfm_init` if "VLFM/视觉" present) |
| No pattern matches | `robot_init` + `robot_get_status` (fallback) |

Grid auto-injection: if the task mentions navigation or exploration keywords
and no grid has been built yet, a `nav_build_grid` call is prepended
automatically using the scene path from the shared world state.

`run_offline(task)` executes the planned sequence through `dispatch_tool`,
stopping on the first failure.

### Live mode

`codex_adapter.py` provides Codex SDK integration.  When `--mode live` is used,
the adapter:

1. Constructs a `CodexConfig` with the `strech-codex-server` MCP server
   registered as a required stdio server.
2. Starts a Codex thread with `sandbox=workspace_write`.
3. Sends a prompt that enumerates all 31 tools and their semantics.
4. Streams events back to the CLI for real-time display (`--verbose`).

The prompt instructs the LLM to:
- Use only the `strech-codex-server` MCP tools.
- Call `nav_build_grid` before any path planner.
- Call `robot_init` before any robot control tool.
- Report errors rather than claiming success when a tool fails.

Each live task prints MCP tool start/completion events and creates a timestamped
episode directory under `src/strech_codex/logs/`. The directory contains a
same-name JSON log and, for robot tasks, an MP4 covering `robot_init` through
`robot_stop` as a labelled mosaic of every available RGB/depth camera. For
state-changing actions (`nav_execute_path`, joint/base commands, home/stow and
grasp/release), `evidence/<sequence>_<tool>/{start,end}/` stores every available
RGB view as PNG, raw depth as NPY, a colourized depth preview, and
`metadata.json` with timestamp, robot state, tool, and phase. The JSON
`artifacts` field links the MP4 and evidence files. Sensitive fields such as
API keys and tokens are redacted.

## CLI (`cli.py`)

```
usage: python -m strech_codex.cli [-h] [--mode {offline,inspect,live}]
                                  [--json] [--robot ROBOT] [--scene SCENE]
                                  [--base-url BASE_URL] [--model MODEL]
                                  [--verbose]
                                  [task]

Robot Codex — MCP-based orchestration for MuJoCo robots
```

| Flag | Description |
|---|---|
| `--mode offline` | Deterministic local planner (default). No API key needed. |
| `--mode live` | Codex SDK orchestration. Requires `[codex]` extras and API key. |
| `--mode inspect` | Print SDK/API availability as JSON. |
| `--json` | Output results as JSON instead of terminal-friendly text. |
| `--robot` | Set robot type: `stretch3`, `google_robot`, `tidybot`. |
| `--scene` | Path to MuJoCo scene XML. Defaults to the bundled `office_scene.xml`. |
| `--base-url` | API base URL for live mode (or set `OPENAI_BASE_URL`). |
| `--model` | Model name for live mode (or set `CODEX_MODEL`). |
| `--verbose` | Stream Codex events in real time during live mode. |

### Examples

```bash
# Navigation
.venv/bin/python -m strech_codex.cli "从 (-0.7, -2.7) 导航到 (2.2, -0.5)"

# Robot control
.venv/bin/python -m strech_codex.cli --robot stretch3 "启动机器人并抬起手臂"

# JSON output
.venv/bin/python -m strech_codex.cli --json "从 (0, 0) 到 (1, 1)"

# Custom scene
.venv/bin/python -m strech_codex.cli --scene my_scene.xml "构建地图"

# Live with custom model
.venv/bin/python -m strech_codex.cli --mode live --model deepseek-chat \
    --base-url https://api.deepseek.com/v1 "探索房间并找到椅子"
```

## MCP server (`mcp_server.py`)

The MCP server can be run as a stdio process for any MCP-compatible client
(Codex CLI, Claude Code, etc.).

```bash
# Direct launch (requires mcp SDK).
.venv/bin/python -m strech_codex.mcp_server

# Register with Codex CLI.
codex mcp add strech-codex -- .venv/bin/python -m strech_codex.mcp_server
```

### Import usage

```python
from strech_codex.mcp_server import dispatch_tool, tool_manifest

# Inspect available tools.
for tool in tool_manifest():
    print(tool["name"], "-", tool["description"])

# Call a tool directly.
result = dispatch_tool("nav_astar", {
    "start_x": 0.0, "start_y": 0.0,
    "goal_x": 2.0, "goal_y": 1.0,
})
print(result["waypoints"])
```

## Supported robots

| Capability | Stretch 3 | Google Robot | TidyBot |
|---|---|---|---|
| `move_to` / `move_by` | ✓ | ✓ | ✓ |
| Base | Differential | Planar compatibility | Planar / omni |
| Lateral movement | ✗ | ✓ | ✓ |
| Actuator types | Position + Velocity | Position | Position + General |
| RGB-D cameras | D405, D435i, Nav | Head RGB + sim depth | Base/wrist RGB + sim depth |
| Internal sensors | IMU + lidar | 2×IMU + ToF + cliff | None |
| Grasp frame | `link_grasp_center` | `gripper` site | `pinch_site` |
| Keyframes | `home`, `stow` | None | `home`, `retract` |
| Runtime model | Multiprocess | Physics thread | Physics thread |

Camera selection in `robot_init`:

```python
# Enable all RGB cameras
robot_init("stretch3", cameras_to_use=["all_rgb"])

# Enable specific cameras by enum name
robot_init("stretch3", cameras_to_use=["d405_rgb", "d405_depth"])

# Enable all cameras
robot_init("stretch3", cameras_to_use=["all"])
```

## Navigation algorithms

The navigation tools wrap the `stretch_mujoco.navigations` package.  All four
algorithms operate on a 2-D occupancy grid rasterised from MuJoCo collision
geometry.

### Occupancy grid

Built by `nav_build_grid` from a MuJoCo scene:

- **Rasterisation:** For each collision geom, world-aligned bounding boxes are
  computed and discretised into grid cells.
- **Inflation:** Each obstacle is inflated by `agent_radius` metres.
- **Height filter:** Geoms entirely below `minimum_obstacle_height` (floor) or
  above `maximum_obstacle_height` (ceiling) are ignored.
- **Mocap exclusion:** Bodies driven by mocap (animated humanoids) are excluded.
- **Collision filter:** When `require_collision=True` (default), only geoms with
  `contype` and `conaffinity` set are treated as obstacles.  Set to `False` for
  visual-only scenes (e.g. Habitat imports).

### A\* vs FMM

| Property | A\* | FMM |
|---|---|---|
| Search direction | Start → goal | Start → all (wavefront) |
| Path extraction | Backtracking via `came_from` | Gradient descent on arrival times |
| Best for | Single queries | Multiple queries on same grid |
| Smoothing | Greedy line-of-sight shortcut | Same |

Both algorithms produce equivalent-quality paths; A\* is typically faster for
single queries, while FMM excels when many queries share the same start point.

### FBE (Frontier-Based Exploration)

Classic autonomous exploration: the robot maintains a local occupancy map
updated by simulated laser scans, detects frontiers (boundaries between free and
unknown space), clusters them, and navigates to the best frontier using A\*.

Key parameters:
- `num_rays=180`, `max_range_m=5.0`, `fov_degrees=270` — simulated laser config
- `min_cluster_size=3` — minimum frontier cluster size in cells
- `explore_threshold=0.85` — stop when 85% of the area is explored
- `max_stagnant_scans=30` — terminate after 30 scans with no new information

### VLFM (Vision-Language Frontier Maps)

Extends FBE with semantic guidance: at each decision step the robot captures an
egocentric RGB-D image, scores it against a natural-language instruction using a
VLM (CLIP / SigLIP / GPT-4o / BLIP2), and projects the score through the depth
FOV onto a spatial value map.  Frontiers are then ranked by accumulated value
instead of purely geometric heuristics.

VLM backends:

| Backend | `vlm_type` | Requirements |
|---|---|---|
| CLIP | `"clip"` | `pip install strech-codex[vlfm]` |
| SigLIP | `"siglip"` | `pip install strech-codex[vlfm]` |
| OpenAI GPT-4o | `"openai"` | `OPENAI_API_KEY` env var, `pip install strech-codex[vlfm]` |
| BLIP2 ITM | `"blip2"` | Separate BLIP2 ITM service at `http://127.0.0.1:12182` |

When `allow_geometric_fallback=True`, VLFM falls back to geometric (FBE-style)
frontier selection if VLM inference fails.

## Adding a new robot

1. Implement the robot adapter in `stretch_mujoco/robots/<name>/` following
   the [robot interface guide](../stretch_mujoco/robots/README.md).
2. Register it in `RobotType` and `create_simulator()` in
   `stretch_mujoco/robots/__init__.py`.
3. No changes needed in `strech_codex` — `robot_init(robot_type="<name>")`
   will pick it up automatically through the factory.

## Adding a new tool

1. Add the tool function to the appropriate module under `tools/` (or create
   a new module).  Every tool function takes keyword arguments and returns
   `{"success": bool, "message": str, ...}`.
2. Register the function in the module's `*_TOOL_FUNCTIONS` dict.
3. Add its metadata to `tool_manifest()` in `mcp_server.py`.
4. Register a `@server.tool()` wrapper in `build_mcp_server()`.
5. If the offline planner should recognise it, add a pattern to `plan_offline()`
   in `agent.py`.

## Adding a new navigation algorithm

1. Subclass `BasePlanner` in `stretch_mujoco/navigations/` and implement
   `plan(grid, start, goal)`.
2. Register it via `register_planner(Algorithm("my_algo"), MyPlanner)`.
3. Add a tool function in `tools/navigation.py` that creates the planner
   and calls `plan()` on the current grid.
4. Register the tool in `NAV_TOOL_FUNCTIONS` and `tool_manifest()`.

## Testing

```bash
# All tests (skip slow integration tests).
.venv/bin/python -m pytest strech_codex/tests/ -q -k "not slow"

# Include slow tests (full robot init/stop lifecycle).
.venv/bin/python -m pytest strech_codex/tests/ -q

# Specific test files.
.venv/bin/python -m pytest strech_codex/tests/test_navigation_tools.py -q -v
.venv/bin/python -m pytest strech_codex/tests/test_robot_tools.py -q -v
.venv/bin/python -m pytest strech_codex/tests/test_agent_cli.py -q -v
.venv/bin/python -m pytest strech_codex/tests/test_codex_adapter.py -q -v
```

Test inventory:

| File | Tests | What it covers |
|---|---|---|
| `test_navigation_tools.py` | 16 | Grid build, A\*/FMM planning, endpoint handling, three drive-type followers, start guard |
| `test_robot_tools.py` | 10 + 5 slow | Lifecycle, cameras, scene fallback, and real path execution on all three robots |
| `test_agent_cli.py` | 4 | Default scene exists, `--robot` feeds planner, joint move regex, malformed coordinate handling |
| `test_codex_adapter.py` | 1 | MCP server registration in Codex config |
| `test_video_recorder.py` | 1 | MP4 recording and episode artifact linkage |
| `test_navigation_execution_integration.py` | 1 slow | Google Robot full-scene A\* plan and execution |

## Reference

- [stretch_mujoco robot interface](../stretch_mujoco/robots/README.md) — unified
  `RobotSimulator` ABC and factory documentation.
- [rai-opensource/vlfm](https://github.com/rai-opensource/vlfm) — upstream VLFM
  (ICRA 2024).
