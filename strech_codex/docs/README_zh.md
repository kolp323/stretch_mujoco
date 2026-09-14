# strech-codex

基于 Codex 风格的智能代理与 MCP 工具集，通过自然语言控制 MuJoCo 中的机器人。

`strech_codex` 将 `stretch_mujoco` 的机器人抽象层与导航算法封装为 MCP
（Model Context Protocol）工具服务器，并提供一个代理循环，将自然语言指令
转换为工具调用序列——支持离线模式（确定性解析器）与在线模式（Codex/LLM 驱动）。

## 快速开始

```bash
# 在 stretch_mujoco 仓库根目录下，使用项目的虚拟环境。
.venv/bin/python -m pip install -e './strech_codex[mcp,codex]'

# 离线模式 — 确定性工具执行（无需 API key）。
.venv/bin/python -m strech_codex.cli --mode offline "启动 stretch3 机器人并查询状态"

# 启动 MCP stdio 服务器（需要 mcp SDK）。
.venv/bin/python -m strech_codex.mcp_server

# 在线模式 — 通过 Codex SDK 进行 LLM 驱动编排（需要 API key）。
.venv/bin/python -m strech_codex.cli --mode live --verbose "去厨房帮我拿一个苹果"
```

## 架构

```text
                         ┌──────────────────────────┐
                         │   自然语言任务              │
                         │   (NPC / 用户 / 外部指令)    │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │     CLI  (cli.py)         │
                         │  --mode offline|live      │
                         └──────┬──────────┬─────────┘
                                │          │
                   ┌────────────▼──┐  ┌───▼──────────────┐
                   │  agent.py     │  │  codex_adapter.py │
                   │  (确定性        │  │  (Codex SDK /     │
                   │   NL 解析器)    │  │   LLM 编排)       │
                   └──────┬────────┘  └───┬──────────────┘
                          │               │
                          └───────┬───────┘
                                  │ dispatch_tool(name, args)
                         ┌────────▼──────────────────────┐
                         │      mcp_server.py             │
                         │  tool_manifest() → 31 个工具    │
                         │  build_mcp_server() → FastMCP  │
                         └──┬──────────┬──────────┬──────┘
                            │          │          │
               ┌────────────▼──┐ ┌─────▼──────┐ ┌▼──────────┐
               │ tools/        │ │ tools/     │ │ tools/     │
               │ navigation.py │ │ robot.py   │ │ scene.py   │
               │ (A*,FMM,FBE,  │ │ (3 种机器人)│ │ (物体查询)   │
               │  VLFM)        │ │            │ │            │
               └───────┬───────┘ └─────┬──────┘ └─────┬──────┘
                       │               │              │
               ┌───────▼───────────────▼──────────────▼──────┐
               │           world/state.py                    │
               │  共享单例: sim, nav, explorer,               │
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

### 核心设计决策

- **共享世界状态** (`world/state.py`) — 模块级单例持有当前的 `RobotSimulator`、
  `NavigationController` 和探索器实例。所有 MCP 工具函数都从这个单例读取和写入，
  因此 agent 组合的多个工具操作的是同一个 MuJoCo 进程和占据栅格。
- **工具 → MCP 分发** — 每个工具都是一个普通的 Python 函数，返回
  `{"success": bool, "message": str, ...}`。MCP 服务器将它们注册到 `FastMCP`
  实例上，agent 通过 `dispatch_tool(name, arguments)` 直接调用。
- **三个工具组** — 导航（A\*、FMM、FBE、VLFM）、机器人控制（Stretch 3 / Google
  Robot / TidyBot）和场景查询。每组是 `tools/` 下的一个独立模块。
- **离线 + 在线双模式** — `agent.py` 中的确定性解析器可以在没有 LLM 的情况下
  测试工具编排；`codex_adapter.py` 提供 Codex SDK 路径用于真正的 LLM 驱动执行。

## 安装

```bash
# 核心（离线 agent、MCP 服务器 — 无需 LLM 依赖）。
.venv/bin/python -m pip install -e './strech_codex'

# 含 MCP SDK（stdio 服务器和在线模式所需）。
.venv/bin/python -m pip install -e './strech_codex[mcp]'

# 含 Codex SDK（在线模式所需）。
.venv/bin/python -m pip install -e './strech_codex[codex]'

# 含 VLM 后端（VLFM 探索所需）。
.venv/bin/python -m pip install -e './strech_codex[vlfm]'

# 开发依赖。
.venv/bin/python -m pip install -e './strech_codex[dev]'
```

## 目录结构

```text
strech_codex/
├── pyproject.toml
├── README.md
├── docs/
│   └── README_zh.md          # 本文档（中文版）
├── src/strech_codex/
│   ├── __init__.py
│   ├── agent.py              # 离线 NL → 工具调用规划器
│   ├── cli.py                # CLI 入口
│   ├── codex_adapter.py      # Codex SDK 在线模式集成
│   ├── mcp_server.py         # FastMCP 服务器 + 分发层
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── navigation.py     # A*、FMM、FBE、VLFM 工具
│   │   ├── robot.py          # 机器人控制工具
│   │   └── scene.py          # 场景查询工具
│   └── world/
│       ├── __init__.py
│       └── state.py          # 共享世界状态单例
└── tests/
    ├── test_navigation_tools.py   # 栅格构建、A*、FMM
    ├── test_robot_tools.py        # 机器人生命周期、状态、场景回退
    ├── test_agent_cli.py          # CLI 配置、规划器回归
    └── test_codex_adapter.py      # 在线模式 MCP 服务器注册
```

## 世界状态 (`world/state.py`)

模块级 `_WorldState` 单例持有跨工具调用共享的所有可变状态。
此模式参考了 `robot_project/src/robot_project/robot_tools.py`。

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

公有访问器（供工具函数使用）：

| 函数 | 用途 |
|---|---|
| `get_sim()` / `set_sim()` | 当前 `RobotSimulator` |
| `has_sim()` | 机器人工具的前置检查 |
| `get_nav()` / `set_nav()` | 当前 `NavigationController` |
| `has_nav()` | 导航工具的前置检查 |
| `get_explorer()` / `set_explorer()` | 当前 `FBEPlanner` 或 `VLFMPlanner` |
| `has_explorer()` | 探索工具的前置检查 |
| `get_scene_path()` / `set_scene_path()` | 当前 MuJoCo 场景 XML 路径 |
| `get_robot_type()` / `set_robot_type()` | `"stretch3"`、`"google_robot"` 或 `"tidybot"` |
| `reset_world()` | 停止仿真器，清除所有状态 |

## MCP 工具（共 31 个）

### 导航工具（12 个）

#### 栅格管理

| 工具 | 参数 | 描述 |
|---|---|---|
| `nav_build_grid` | `scene_xml?`, `bounds?`, `resolution=0.08`, `agent_radius=0.25`, `floor_geom_name="office_floor"`, `minimum_obstacle_height=0.08`, `maximum_obstacle_height=1.80`, `require_collision=True`, `exclude_prefixes?` | 将 MuJoCo 碰撞几何体栅格化为 2-D 占据栅格。必须在任何路径规划器之前调用。返回栅格尺寸、空闲单元数和边界。 |
| `nav_get_grid_info` | _(无)_ | 返回当前栅格的尺寸、分辨率、边界和空闲比例。 |
| `nav_is_free` | `x`, `y` | 检查世界坐标点是否在当前栅格上可通行。 |

#### A\*（A-star）

| 工具 | 参数 | 描述 |
|---|---|---|
| `nav_astar` | `start_x`, `start_y`, `goal_x`, `goal_y`, `smoothing=True` | 在 8 连通栅格上使用 A\* 规划无碰撞路径。返回有序路径点列表、路径距离和路径点数量。 |

**算法细节：** 标准 A\* 搜索，使用欧几里得距离启发式、8 连通邻域（对角线代价
`√2`）、角点切割防护，以及贪心路径平滑——当两点间直线无碰撞时移除中间路径点。

#### FMM（快速行进法）

| 工具 | 参数 | 描述 |
|---|---|---|
| `nav_fmm` | `start_x`, `start_y`, `goal_x`, `goal_y`, `smoothing=True` | 从起点向外求解 Eikonal 方程 `|∇T| = 1`，然后通过梯度下降从目标点提取路径。 |

**算法细节：** 均匀栅格上的一阶迎风 FMM 离散化。到达时间场从起点向外计算，
直到目标单元被接受。梯度下降沿最陡时间下降方向行进，每步验证视线是否无碰撞。

#### 路径执行

| 工具 | 参数 | 描述 |
|---|---|---|
| `nav_execute_path` | `waypoints`, `timeout=120`, `position_tolerance=0.12`, `start_tolerance=0.5`, `max_linear_speed=0.4`, `max_angular_speed=0.8`, `control_hz=20` | 使用闭环底盘控制跟踪世界坐标路径点。Stretch 3 使用差速控制，Google Robot/TidyBot 使用全向控制；返回前始终发送零速度。 |

#### FBE（Frontier-Based Exploration，基于边界的自主探索）

| 工具 | 参数 | 描述 |
|---|---|---|
| `nav_fbe_init` | `scene_xml?`, `bounds?`, `resolution=0.08`, `agent_radius=0.25`, `num_rays=180`, `max_range_m=5.0`, `fov_degrees=270.0`, `min_cluster_size=3`, `explore_threshold=0.85`, `start_x=0.0`, `start_y=0.0`, `start_yaw=0.0` | 构建真值栅格并创建带仿真激光扫描仪的 `FBEPlanner`。循环调用 `nav_fbe_step` 进行探索。 |
| `nav_fbe_step` | `robot_x`, `robot_y`, `robot_yaw` | 推进探索一步（~10-20 Hz）。发射射线、更新局部地图、检测边界、聚类、规划 A\* 路径到最佳边界。返回状态、已探索比例、边界数量和下一个移动指令。 |
| `nav_fbe_path` | _(无)_ | 返回当前 FBE 路径的路径点。 |

**状态机：** `SCANNING → PLANNING → MOVING → ROTATING → FINISHED`。
当已探索比例达到阈值且无可达边界时终止，或在配置的停滞扫描次数后终止。

#### VLFM（Vision-Language Frontier Maps，视觉-语言边界地图）

| 工具 | 参数 | 描述 |
|---|---|---|
| `nav_vlfm_init` | `instruction`, `scene_xml?`, `vlm_type="clip"`, `device="cpu"`, `resolution=0.08`, `agent_radius=0.25`, `max_range_m=5.0`, `explore_threshold=0.85`, `allow_geometric_fallback=True`, `start_x/y/yaw` | 用自然语言指令（如"前面似乎有一把椅子"）初始化 VLFM 探索。构建真值栅格，加载 VLM 模型（CLIP / SigLIP / OpenAI / BLIP2）。 |
| `nav_vlfm_step` | `robot_x`, `robot_y`, `robot_yaw` | 推进 VLFM 一步。通过累积价值地图评分边界，强制执行无环选择以防止振荡，然后规划 A\* 到最佳边界。 |
| `nav_vlfm_inject_observation` | `rgb_image_path`, `depth_image_path`, `camera_x/y/yaw`, `hfov_rad=1.2`, `min_depth=0.1`, `max_depth=5.0` | 将 RGB 图像与 VLFM 指令进行评分，然后通过深度图像的相机视场角将评分投影到空间价值地图上。 |

**算法细节：** 对齐上游 [rai-opensource/vlfm](https://github.com/rai-opensource/vlfm)
ITM（Image-Text Matching）策略模式（ICRA 2024）。在每个决策步骤，VLM 对当前
自我中心视图进行评分；评分通过深度 FOV 投影到空间价值地图上。边界按累积价值
地图评分排序，`AcyclicEnforcer` 防止重复访问最近探索过的区域。双通道模式
（目标 + 探索）根据可配置的阈值在"寻找"和"探索"之间切换。

### 机器人控制工具（16 个）

#### 生命周期

| 工具 | 参数 | 描述 |
|---|---|---|
| `robot_init` | `robot_type`, `scene_xml?`, `headless=True`, `cameras_to_use?`, `camera_hz=30.0`, `start_translation?`, `start_rotation_quat?` | 创建并启动机器人仿真器。`robot_type` 为 `"stretch3"`、`"google_robot"` 或 `"tidybot"`。`cameras_to_use` 接受相机枚举名称或字符串别名 `"all_rgb"`、`"all_depth"`、`"all"`。 |
| `robot_stop` | _(无)_ | 停止仿真器并从共享世界中清除机器人状态。 |

#### 关节控制

| 工具 | 参数 | 描述 |
|---|---|---|
| `robot_move_to` | `actuator_name`, `position` | 指令绝对关节位置目标。通过枚举名称、枚举值或 MJCF 关节名称解析执行器。 |
| `robot_move_by` | `actuator_name`, `delta` | 指令相对位置增量。 |
| `robot_set_base_velocity` | `v_linear=0.0`, `omega=0.0`, `v_lateral=0.0` | 设置移动底盘速度。`v_lateral` 仅适用于全向底盘（Google Robot、TidyBot）；Stretch 3 会抛出 `ValueError`。 |
| `robot_home` | _(无)_ | 移动到 home 关键帧（Stretch 3、TidyBot；Google Robot 抛出 `NotImplementedError`）。 |
| `robot_stow` | _(无)_ | 移动到 stow/retract 关键帧。 |
| `robot_wait_until_at_setpoint` | `actuator_name`, `timeout=5.0`, `position_tolerance=0.05` | 阻塞直到执行器到达其指令的 `move_to` 目标。 |

#### 状态查询

| 工具 | 参数 | 描述 |
|---|---|---|
| `robot_get_status` | _(无)_ | 拉取完整关节状态快照（dict 格式）。字段因机器人类型而异。 |
| `robot_get_base_pose` | _(无)_ | 返回世界坐标系中的 `(x, y, theta)`。 |
| `robot_get_ee_pose` | _(无)_ | 返回 4×4 末端执行器位姿矩阵。 |
| `robot_get_camera_data` | `camera_name=""` | 提供 `camera_name` 时返回该相机的图像元数据（shape, dtype）。否则列出可用的 RGB 和深度相机。 |

#### 抓取

| 工具 | 参数 | 描述 |
|---|---|---|
| `robot_attach_object` | `object_id` | 将指定物体固定到夹爪上进行稳定抓取仿真。 |
| `robot_release_object` | _(无)_ | 释放当前附着的物体。 |

#### 内省

| 工具 | 参数 | 描述 |
|---|---|---|
| `robot_list_actuators` | _(无)_ | 列出所有执行器，含名称、值、类型（`POSITION`/`VELOCITY`/`GENERAL`）以及是否为底盘执行器。 |
| `robot_list_cameras` | _(无)_ | 列出所有相机，含名称、MJCF 名称、RGB/深度分类。 |

### 场景工具（3 个）

| 工具 | 参数 | 描述 |
|---|---|---|
| `scene_list_objects` | _(无)_ | 枚举场景中的自由关节体（可移动物体）。如果仿真器后端未直接暴露 `mjmodel`，则回退到场景 XML。 |
| `scene_get_object_pose` | `object_name` | 返回指定物体的 4×4 世界位姿。 |
| `scene_add_world_frame` | `x=0.0`, `y=0.0`, `z=0.0` | 在指定世界坐标处添加坐标系标记（查看器可视化）。 |

## Agent 系统 (`agent.py`)

### 离线规划器

`plan_offline(task)` 使用正则表达式模式将中英文自然语言指令解析为 `ToolCall`
序列。它能识别以下模式：

| 模式 | 生成的工具调用 |
|---|---|
| "构建地图" / "build grid" | `nav_build_grid` |
| "从 (sx, sy) 到 (gx, gy)" / "导航到 (x, y)" | `nav_astar`（若无栅格则自动注入 `nav_build_grid`） |
| "启动 stretch3 / google_robot / tidybot" | `robot_init` |
| "移动 lift 到 0.5" / "move arm to 1.0" | `robot_move_to` |
| "前进" / "后退" / "左转" / "右转" | `robot_set_base_velocity` |
| "抬起手臂" / "放下手臂" | `robot_move_to(lift, …)` |
| "抓取/拿 物体" / "释放" | `robot_attach_object` / `robot_release_object` |
| "状态" / "位置" / "在哪" | `robot_get_status` + `robot_get_base_pose` |
| "探索" / "扫描" | `nav_fbe_init`（如有"VLFM/视觉"则为 `nav_vlfm_init`） |
| 无模式匹配 | `robot_init` + `robot_get_status`（回退） |

栅格自动注入：如果任务包含导航或探索关键词且尚未构建栅格，则使用共享世界
状态中的场景路径自动在前面插入 `nav_build_grid` 调用。

`run_offline(task)` 通过 `dispatch_tool` 执行规划好的序列，遇到首个失败即停止。

### 在线模式

`codex_adapter.py` 提供 Codex SDK 集成。使用 `--mode live` 时，适配器：

1. 构建 `CodexConfig`，将 `strech-codex-server` MCP 服务器注册为必需的 stdio 服务器。
2. 以 `sandbox=workspace_write` 启动 Codex 线程。
3. 发送包含全部 31 个工具及其语义的提示词。
4. 将事件流式传回 CLI 进行实时显示（`--verbose`）。

提示词要求 LLM：
- 只使用 `strech-codex-server` MCP 工具。
- 在任何路径规划器之前调用 `nav_build_grid`。
- 在任何机器人控制工具之前调用 `robot_init`。
- 工具失败时报告错误，而非声称成功。

每个在线任务会在 `src/strech_codex/logs/任务名_时间戳/` 生成同名 JSON 和
MP4。MP4 从 `robot_init` 持续记录到 `robot_stop`，以带标签的拼图展示该机器人
所有实际可用的 RGB/深度相机。导航执行、关节/底盘控制、home/stow、抓取/释放等
状态变更工具还会在 `evidence/序号_工具名/{start,end}/` 保存动作前后证据：RGB
PNG、原始深度 NPY、彩色深度预览 PNG，以及包含时间、机器人状态、工具和阶段的
`metadata.json`。主 episode JSON 的 `artifacts` 会索引这些文件。

## CLI (`cli.py`)

```
usage: python -m strech_codex.cli [-h] [--mode {offline,inspect,live}]
                                  [--json] [--robot ROBOT] [--scene SCENE]
                                  [--base-url BASE_URL] [--model MODEL]
                                  [--verbose]
                                  [task]

Robot Codex — MCP-based orchestration for MuJoCo robots
```

| 参数 | 描述 |
|---|---|
| `--mode offline` | 确定性本地规划器（默认）。无需 API key。 |
| `--mode live` | Codex SDK 编排。需要 `[codex]` 额外依赖和 API key。 |
| `--mode inspect` | 以 JSON 格式打印 SDK/API 可用性。 |
| `--json` | 以 JSON 格式输出结果，替代终端友好的文本格式。 |
| `--robot` | 设置机器人类型：`stretch3`、`google_robot`、`tidybot`。 |
| `--scene` | MuJoCo 场景 XML 路径。默认使用内置的 `office_scene.xml`。 |
| `--base-url` | 在线模式的 API 地址（或设置 `OPENAI_BASE_URL`）。 |
| `--model` | 在线模式的模型名称（或设置 `CODEX_MODEL`）。 |
| `--verbose` | 在线模式下实时流式输出 Codex 事件。 |

### 示例

```bash
# 导航
.venv/bin/python -m strech_codex.cli "从 (-0.7, -2.7) 导航到 (2.2, -0.5)"

# 机器人控制
.venv/bin/python -m strech_codex.cli --robot stretch3 "启动机器人并抬起手臂"

# JSON 输出
.venv/bin/python -m strech_codex.cli --json "从 (0, 0) 到 (1, 1)"

# 自定义场景
.venv/bin/python -m strech_codex.cli --scene my_scene.xml "构建地图"

# 在线模式使用自定义模型
.venv/bin/python -m strech_codex.cli --mode live --model deepseek-chat \
    --base-url https://api.deepseek.com/v1 "探索房间并找到椅子"
```

## MCP 服务器 (`mcp_server.py`)

MCP 服务器可以作为 stdio 进程运行，供任何兼容 MCP 的客户端使用
（Codex CLI、Claude Code 等）。

```bash
# 直接启动（需要 mcp SDK）。
.venv/bin/python -m strech_codex.mcp_server

# 注册到 Codex CLI。
codex mcp add strech-codex -- .venv/bin/python -m strech_codex.mcp_server
```

### 通过 import 使用

```python
from strech_codex.mcp_server import dispatch_tool, tool_manifest

# 查看可用工具。
for tool in tool_manifest():
    print(tool["name"], "-", tool["description"])

# 直接调用工具。
result = dispatch_tool("nav_astar", {
    "start_x": 0.0, "start_y": 0.0,
    "goal_x": 2.0, "goal_y": 1.0,
})
print(result["waypoints"])
```

## 支持的机器人

| 能力 | Stretch 3 | Google Robot | TidyBot |
|---|---|---|---|
| `move_to` / `move_by` | ✓ | ✓ | ✓ |
| 底盘 | 差速 | Planar 兼容 | Planar / 全向 |
| 横向移动 | ✗ | ✓ | ✓ |
| 执行器类型 | Position + Velocity | Position | Position + General |
| RGB-D 相机 | D405, D435i, Nav | 头部 RGB + 仿真深度 | 底座/腕部 RGB + 仿真深度 |
| 内部传感器 | IMU + 激光雷达 | 2×IMU + ToF + 悬崖 | 无 |
| 抓取坐标系 | `link_grasp_center` | `gripper` site | `pinch_site` |
| 关键帧 | `home`, `stow` | 无 | `home`, `retract` |
| 运行模型 | 多进程 | 物理线程 | 物理线程 |

`robot_init` 中的相机选择：

```python
# 启用所有 RGB 相机
robot_init("stretch3", cameras_to_use=["all_rgb"])

# 按枚举名称启用特定相机
robot_init("stretch3", cameras_to_use=["d405_rgb", "d405_depth"])

# 启用所有相机
robot_init("stretch3", cameras_to_use=["all"])
```

## 导航算法

导航工具封装了 `stretch_mujoco.navigations` 包。全部四种算法均基于从 MuJoCo
碰撞几何体栅格化得到的 2-D 占据栅格运行。

### 占据栅格

由 `nav_build_grid` 从 MuJoCo 场景构建：

- **栅格化：** 对每个碰撞几何体，计算世界坐标系对齐的包围盒并离散化为栅格单元。
- **膨胀：** 每个障碍物向外膨胀 `agent_radius` 米。
- **高度过滤：** 完全低于 `minimum_obstacle_height`（地面）或高于
  `maximum_obstacle_height`（天花板）的几何体被忽略。
- **动作捕捉排除：** 由 mocap 驱动的物体（动画人物）被排除。
- **碰撞过滤：** 当 `require_collision=True`（默认）时，仅设置了 `contype` 和
  `conaffinity` 的几何体被视为障碍物。对于仅有视觉的 Habitat 场景设置为 `False`。

### A\* vs FMM

| 特性 | A\* | FMM |
|---|---|---|
| 搜索方向 | 起点 → 目标 | 起点 → 全部（波前） |
| 路径提取 | 通过 `came_from` 回溯 | 在到达时间场上进行梯度下降 |
| 最适合 | 单次查询 | 同一栅格上的多次查询 |
| 平滑 | 贪心视线捷径 | 相同 |

两种算法产出的路径质量相当；对于单次查询 A\* 通常更快，而 FMM 在多个查询
共享同一起点时表现优异。

### FBE（Frontier-Based Exploration，基于边界的自主探索）

经典自主探索：机器人维护由仿真激光扫描更新的局部占据地图，检测边界（空闲与
未知空间之间的边界），将其聚类，并使用 A\* 导航到最佳边界。

关键参数：
- `num_rays=180`, `max_range_m=5.0`, `fov_degrees=270` — 仿真激光配置
- `min_cluster_size=3` — 最小边界聚类大小（单元数）
- `explore_threshold=0.85` — 当 85% 的区域被探索后停止
- `max_stagnant_scans=30` — 连续 30 次扫描无新信息后终止

### VLFM（Vision-Language Frontier Maps，视觉-语言边界地图）

在 FBE 基础上扩展语义引导：在每个决策步骤，机器人捕获自我中心 RGB-D 图像，
使用 VLM（CLIP / SigLIP / GPT-4o / BLIP2）对其与自然语言指令进行评分，
并通过深度 FOV 将评分投影到空间价值地图上。然后按累积价值而非纯几何启发式
对边界进行排序。

VLM 后端：

| 后端 | `vlm_type` | 依赖要求 |
|---|---|---|
| CLIP | `"clip"` | `pip install strech-codex[vlfm]` |
| SigLIP | `"siglip"` | `pip install strech-codex[vlfm]` |
| OpenAI GPT-4o | `"openai"` | `OPENAI_API_KEY` 环境变量, `pip install strech-codex[vlfm]` |
| BLIP2 ITM | `"blip2"` | 独立的 BLIP2 ITM 服务，位于 `http://127.0.0.1:12182` |

当 `allow_geometric_fallback=True` 时，如 VLM 推理失败，VLFM 回退到几何
（FBE 风格）边界选择。

## 添加新机器人

1. 参照[机器人接口指南](../stretch_mujoco/robots/README.md)，在
   `stretch_mujoco/robots/<name>/` 中实现机器人适配器。
2. 在 `stretch_mujoco/robots/__init__.py` 的 `RobotType` 和 `create_simulator()`
   中注册。
3. `strech_codex` 无需任何修改 — `robot_init(robot_type="<name>")`
   将通过工厂自动获取。

## 添加新工具

1. 将工具函数添加到 `tools/` 下的相应模块（或创建新模块）。每个工具函数接受
   关键字参数并返回 `{"success": bool, "message": str, ...}`。
2. 在该模块的 `*_TOOL_FUNCTIONS` 字典中注册该函数。
3. 在 `mcp_server.py` 的 `tool_manifest()` 中添加其元数据。
4. 在 `build_mcp_server()` 中注册 `@server.tool()` 包装器。
5. 若离线规划器应识别该工具，在 `agent.py` 的 `plan_offline()` 中添加模式。

## 添加新导航算法

1. 在 `stretch_mujoco/navigations/` 中子类化 `BasePlanner`，实现
   `plan(grid, start, goal)`。
2. 通过 `register_planner(Algorithm("my_algo"), MyPlanner)` 注册。
3. 在 `tools/navigation.py` 中添加工具函数，创建该规划器并在当前栅格上调用 `plan()`。
4. 在 `NAV_TOOL_FUNCTIONS` 和 `tool_manifest()` 中注册该工具。

## 测试

```bash
# 全部测试（跳过慢速集成测试）。
.venv/bin/python -m pytest strech_codex/tests/ -q -k "not slow"

# 包含慢速测试（完整的机器人 init/stop 生命周期）。
.venv/bin/python -m pytest strech_codex/tests/ -q

# 单独测试文件。
.venv/bin/python -m pytest strech_codex/tests/test_navigation_tools.py -q -v
.venv/bin/python -m pytest strech_codex/tests/test_robot_tools.py -q -v
.venv/bin/python -m pytest strech_codex/tests/test_agent_cli.py -q -v
.venv/bin/python -m pytest strech_codex/tests/test_codex_adapter.py -q -v
```

测试清单：

| 文件 | 测试数 | 覆盖内容 |
|---|---|---|
| `test_navigation_tools.py` | 16 | 栅格构建、A\*/FMM 规划、端点处理、三种底盘跟踪器、起点保护 |
| `test_robot_tools.py` | 10 + 5 slow | 生命周期、相机、场景回退、三台真实机器人的路径执行 |
| `test_agent_cli.py` | 4 | 默认场景存在性、`--robot` 输入到规划器、关节移动正则、畸形坐标处理 |
| `test_codex_adapter.py` | 1 | Codex 配置中的 MCP 服务器注册 |
| `test_video_recorder.py` | 1 | MP4 录像及 episode artifact 关联 |
| `test_navigation_execution_integration.py` | 1 slow | Google Robot 全场景 A\* 规划与执行 |

## 参考

- [stretch_mujoco 机器人接口](../stretch_mujoco/robots/README.md) — 统一的
  `RobotSimulator` ABC 与工厂文档。
- [rai-opensource/vlfm](https://github.com/rai-opensource/vlfm) — 上游 VLFM
  （ICRA 2024）。
