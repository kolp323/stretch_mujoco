# Stretch MuJoCo 项目说明

> 本文基于当前工作区代码、配置和已有文档整理，面向第一次接触项目的开发者。当前包版本为 `0.5.0`，工作区检出分支为 `feat/npc-system`。

## 一、用一句话理解项目

这是一个使用 MuJoCo 在电脑中搭建 Stretch 机器人及其工作环境的 Python 仿真平台。

它不只是显示机器人模型，还提供了机器人控制、相机和传感器读取、办公/厨房场景、导航、抓取、NPC、人类员工 Agent、语义世界，以及可选的 Codex/MCP 工具层：

```text
场景 XML / 3D 资产
        ↓
MuJoCo 物理仿真
        ↓
Stretch 机器人控制与传感器接口
        ↓
导航、抓取、NPC、办公智能体、LLM/Codex
```

## 二、项目能力总览

| 能力 | 简单说明 | 主要位置 |
| --- | --- | --- |
| MuJoCo 仿真 | 加载 XML、物理步进、碰撞、渲染和 Viewer | `stretch_mujoco/mujoco_server*.py` |
| Stretch 3 控制 | 底盘、升降、机械臂、腕部、夹爪和头部 | `stretch_mujoco/stretch_mujoco_simulator.py`、`robots/stretch3/` |
| 多机器人抽象 | 统一接口支持 Stretch 3、Google Robot、TidyBot | `stretch_mujoco/robots/` |
| 视觉/传感器 | RGB、深度、相机内参、IMU、激光测距 | `datamodels/`、相机/传感器 manager |
| 场景和资产 | 办公室、零食、基础模型、Robocasa 厨房 | `models/`、`robocasa_gen.py` |
| 2D 导航 | A\*、FMM、FBE、VLFM、NavDP、InternVLA、QwenS2 | `stretch_mujoco/navigations/` |
| 抓取 | 抓取任务、标定、IK、物体附着和结果指标 | `grasp_task.py`、`graspgen/` |
| 人形 NPC | SMPL-X、动作播放、坐下和导航 | `humanoid/` |
| 办公员工 Agent | 日程、需求、决策、记忆、任务和机器人请求 | `agents/` |
| 语义世界 | 对象、关系、位置、所有权、权限和交互点 | `semantics/world.py` |
| Codex/MCP | 把仿真能力封装成可调用的工具 | `strech_codex/` |

当前主包约有 122 个 Python 文件，35 个示例，11 个工具脚本；`tests/` 和 `strech_codex/tests/` 共 41 个测试文件。模型资产体积较大，包含网格、纹理、XML 和碰撞代理。

## 三、核心架构

### 3.1 高层入口：`StretchMujocoSimulator`

普通业务代码主要使用 `StretchMujocoSimulator`，它负责：

- 启动和停止仿真；
- 发送关节绝对位置、相对移动和底盘速度命令；
- 查询关节状态、传感器、相机和语义状态；
- 查询底盘、末端执行器和 link 的世界坐标；
- 控制 NPC 动画、抓取附着、物体可见性和运动速度；
- 等待关节到位或等待运动结束。

最小示例：

```python
from stretch_mujoco import StretchMujocoSimulator

sim = StretchMujocoSimulator()
sim.start(headless=True)
sim.home()
sim.move_to("lift", 0.8)
sim.set_base_velocity(0.2, 0.0)

status = sim.pull_status()
camera = sim.pull_camera_data()
sensors = sim.pull_sensor_data()
sim.stop()
```

### 3.2 独立 MuJoCo 进程

`sim.start()` 会启动独立的 MuJoCo 服务进程：

1. 主进程创建 `multiprocessing.Manager` 和命令/状态代理；
1. 子进程加载 `MjModel`、`MjData` 并运行物理循环；
1. 主进程通过 `MujocoServerProxies` 写入命令、读取状态；
1. 服务端控制器把命令转换成 MuJoCo actuator 控制量；
1. 相机和传感器 manager 更新数据快照；
1. 停止时设置 stop event，并等待子进程结束。

相关代码：

- `mujoco_server.py`：服务端、物理循环、底盘控制、抓取附着和代理；
- `mujoco_server_passive.py`：Passive Viewer；
- `mujoco_server_managed.py`：Managed Viewer；
- `mujoco_server_camera_manager.py`：相机渲染；
- `mujoco_server_sensor_manager.py`：传感器；
- `datamodels/status_command.py`：进程间命令结构。

三种运行方式是可视化 Viewer、Passive/Managed Viewer，以及不打开窗口的 headless。训练、数据采集和自动化测试优先使用 headless。

### 3.3 目录地图

```text
stretch_mujoco/
├── stretch_mujoco/                 # 主 Python 包
│   ├── stretch_mujoco_simulator.py # 高层模拟器 API
│   ├── mujoco_server*.py            # MuJoCo 服务端和 Viewer
│   ├── robots/                      # 多机器人接口和具体实现
│   ├── datamodels/                  # 命令、关节、相机、传感器数据类
│   ├── enums/                       # actuator/camera/sensor 枚举
│   ├── navigations/                 # 栅格、路径规划和探索
│   ├── graspgen/                    # 抓取、标定、IK、客户端协议
│   ├── humanoid/                    # SMPL-X、动画和 NPC 导航
│   ├── agents/                      # 办公员工 Agent 和 LLM
│   ├── semantics/                   # 语义对象和关系图
│   ├── models/                      # XML、JSON、网格和纹理
│   └── robocasa_gen.py              # Robocasa 场景生成
├── examples/                        # 可运行演示
├── tools/                           # 资产、场景、纹理、评估工具
├── tests/                           # 主项目测试
├── strech_codex/                    # 可选 Codex/MCP agent
├── docs/                            # 教程和开发文档
├── third_party/                     # Robocasa、Robosuite 子模块
├── pyproject.toml                   # 依赖和 uv 配置
└── README.md                        # 项目原始快速开始
```

## 四、机器人与数据接口

### 4.1 Stretch 3

主要 actuator 包括：

- `lift`：升降柱；
- `arm`：伸缩臂；
- `wrist_yaw`、`wrist_pitch`、`wrist_roll`：腕部；
- `gripper`：夹爪；
- `head_pan`、`head_tilt`：头部；
- 底盘的平移/旋转和左右轮速度。

位置关节使用 `move_to()` / `move_by()`，底盘使用 `set_base_velocity(v_linear, omega)`。底盘是连续速度控制，不能用绝对位置控制左右轮。

### 4.2 多机器人抽象

`stretch_mujoco.robots` 提供 `RobotSimulator`、`RobotActuators`、`RobotCameras`、`RobotSensors`、`RobotStatus` 等统一类型，并通过工厂创建机器人：

```python
from stretch_mujoco.robots import RobotType, create_simulator

sim = create_simulator(
    RobotType.STRETCH3,
    scene_xml_path="stretch_mujoco/models/scene.xml",
)
```

当前支持 `stretch3`、`google_robot`、`tidybot`。通用生命周期和控制接口相同，但各机器人具体关节、相机、传感器和模型参数不同。

## 五、场景、XML 与资产

`stretch_mujoco/models/` 中的 XML 是物理世界的主要来源，定义 body、geom、joint、actuator、sensor、camera、site、材质和资源引用。常见文件有：

- `scene.xml`：基础场景；
- `stretch.xml`、`stretch_mj_3.3.0.xml`：Stretch 模型；
- `office_scene.xml`：单 NPC 办公场景；
- `office_scene2_multi_npc.xml`：多 NPC 办公场景；
- `docking_station.xml`：停靠站；
- `robocasa_gen.py`：生成厨房 XML。

视觉 mesh 和碰撞 proxy 可能是两个文件：前者负责外观，后者负责碰撞和效率。修改资产必须同时核对尺寸、原点、旋转、碰撞层和语义绑定。

办公场景通常由三类文件一起描述：

1. XML：几何、物理、机器人和 NPC；
1. `office_*_semantics.json`：对象类型、body/geom 绑定、属性和交互点；
1. `office_*_agents.json`：员工画像、日程、需求、偏好和配置。

## 六、导航

### 6.1 占据栅格

`OccupancyGrid.from_model()` 从 MuJoCo 碰撞几何生成二维占据栅格，再按机器人半径膨胀障碍物。常见默认值为 0.08 m 分辨率和 0.25 m agent radius。支持从 `office_floor` 推断边界、手动指定 bounds、筛选障碍高度、跳过非碰撞 geom，以及世界坐标与 cell 互转。

### 6.2 规划器

- `A*/planner.py`：8 邻域 A\*，带欧氏启发式、防对角切角和路径平滑；
- `FMM/planner.py`：求解到达时间场，再沿梯度提取路径；
- `FBE/`：模拟激光扫描、建立局部地图、寻找 frontier 并探索；
- `VLFM/`：用视觉语言模型为 frontier 评分，并提供几何回退；
- `NavDP/`：调用外部 NavDP 服务，再由轨迹跟踪器执行；
- `InternVLAS2/`、`QwenS2/`：外部视觉语言导航策略接口。

`NavigationController` 把路径点转成底盘闭环控制。规划器回答“去哪、走哪条路”，控制器负责“如何实际走过去”。

## 七、抓取

抓取代码分两层：

- `grasp_task.py`：场景级抓取任务；
- `graspgen/`：物体表示、标定、IK、客户端和通信协议。

服务端支持把物体附着到夹爪、释放物体、设置验证目标并返回抓取指标。办公场景还支持物体可见性以及“取走/恢复”物体，便于模拟零食和文件任务。

抓取是否成功取决于视觉位姿、末端位姿、碰撞体、夹爪范围、交互 site 和语义位置是否一致；只修改视觉 mesh 通常不能解决物理抓取问题。

## 八、NPC 与办公 Agent

### 8.1 人形可视化层

`humanoid/` 负责 SMPL-X 资产转换、动作烘焙、mesh 动画播放和 NPC 导航。当前基础动作包括 `idle`、`walk`、`sit`、`work`、`eat`。坐下动作会先规划到椅子附近的无碰撞接近点，再进行最终坐姿对齐。

### 8.2 员工 Agent 逻辑层

`agents/` 是确定性的办公行为运行时，负责：

- 员工 profile、角色、性格和偏好；
- hunger/thirst/fatigue 等 needs；
- 工作日程、事件和 utility 决策；
- perception、memory、planner、executor；
- 预约、权限、冲突、执行结果；
- 向机器人发出取物、递送和 handover 请求。

动作采用封闭集合，例如 `move_to`、`sit`、`work`、`rest`、`eat`、`drink`、`pick_up`、`put_down`、`request_robot`、`handover`、`attend_meeting`。LLM（如果启用）只提出候选意图或文本，最终动作仍需经过 JSON/枚举校验和运行时规则验证。

## 九、语义世界

MuJoCo 负责“几何和真实位姿”，`SemanticWorld` 负责“名字、关系、权限和任务状态”：

```text
MuJoCo：物体在哪里、姿态是什么、是否碰撞
语义层：物体叫什么、属于谁、能否访问、任务是否完成
```

关系以 `(subject, relation, object)` 存储，例如：

```text
document_report INSIDE storage_cabinet
document_report BELONGS_TO employee_01
employee_01 HOLDS document_report
```

常用接口有 `can_access()`、`add_relation()`、`remove_relation()`、`set_location()`、`pending_requests()`。`sim.pull_semantic_state()` 返回低频、可跨进程传递的对象和交互点世界坐标，可用于导航、抓取、handover 和动画。

## 十、Codex/MCP 扩展

`strech_codex/` 是可选扩展，不是主仿真核心。它把仿真能力封装成 MCP 工具，包括：

- 离线任务解析和工具调用计划；
- 机器人启动、状态、移动、相机和传感器；
- 导航栅格、A\*、FMM、FBE/VLFM 和路径执行；
- 语义世界查询和办公 Agent 操作；
- 证据、事件和视频记录；
- 可选 OpenAI/Codex live adapter。

目录名称是现有约定的 `strech_codex`，不要无必要改名。主项目可脱离 MCP 扩展单独运行。

## 十一、安装与常用运行方式

项目使用 `uv`，要求 Python 3.10+：

```bash
git clone https://github.com/hello-robot/stretch_mujoco --recurse-submodules
cd stretch_mujoco
uv sync --extra dev
uv run launch_sim
```

`pyproject.toml` 固定 MuJoCo `3.2.6`，用于兼容 Robocasa。默认依赖还包括 NumPy、OpenCV、urchin、inputs、pynput 和 openpi-client；其中 openpi-client 配置为仓库旁边 `../openpi/packages/openpi-client` 的 editable path，本地没有该路径时需要按环境补齐或调整配置。

常用示例：

```bash
# 基础、办公室和 NPC
uv run examples/office_scene.py
uv run examples/office_scene.py --headless
uv run examples/office_scene.py --animation-demo
uv run examples/office_agents.py --complete-task
uv run examples/office_autonomy.py --seconds 60
uv run examples/office_multi_npc_day.py --end 18 --step 0.25 --chat-log

# 关节、相机、传感器和手动控制
uv run examples/move_joints.py
uv run examples/camera_feeds.py
uv run examples/laser_scan.py
uv run examples/keyboard_teleop.py
uv run examples/gamepad_teleop.py

# 导航和语义
uv run examples/navigation_viewer.py
uv run examples/fbe_viewer.py
uv run examples/vlfm_demo.py
uv run examples/office_semantics.py --object-id document_report
```

Viewer 运行需要有效图形环境；SSH 运行要配置 X11。机器学习、批处理和自动测试优先使用 headless。Robocasa 需要额外安装 `.[robocasa]`、两个子模块宏配置和厨房资产，且要求 Python 3.10。

## 十二、开发、测试和发布

```bash
uv run pytest -q
uv run pytest -q tests/test_office_scene.py
uv run pre-commit run --all-files
uv build
```

测试覆盖机器人控制、办公场景、语义、导航、抓取、NPC、OpenPI、键盘控制、LLM provider 和 MCP 工具。涉及 Viewer、渲染、资产、SMPL-X 或外部模型的测试可能需要额外环境；普通单元测试不能完全代表视觉效果。

发布流程见 `docs/releasing_to_pypi.md`：更新版本和 changelog，合并发布提交，打 tag，运行 `uv build` 检查包内容，再发布到 PyPI。基础 XML 和 Python 模块可打包，但 Robocasa 子模块和部分本地资产不适合作为 PyPI 内容。

## 十三、当前状态与主要风险

### 已具备

- 独立 MuJoCo 进程、Viewer/headless 模式和高层模拟器 API；
- Stretch 3 的控制、状态、相机和传感器链路；
- 多机器人抽象；
- A\*、FMM、frontier 探索和多个外部视觉导航适配器；
- 办公 XML、语义、员工 Agent、NPC 动画和机器人请求链路；
- 抓取、OpenPI 和 MCP 的协议/适配层；
- 覆盖面较广的 pytest 测试。

### 需要注意

1. 环境依赖较多：MuJoCo 图形、Linux 输入设备、Mac `mjpython`、OpenPI sibling path、Robocasa 和外部 VLM 服务都可能影响运行。
1. `mjmodel/mjdata` 的物理步进和渲染有线程/锁约束，违反时可能导致 MuJoCo 崩溃或数值不稳定。
1. 相机越多越慢，rangefinder 激光也较耗算力，应按需启用。
1. 语义 JSON 的 body/geom 名称必须和 XML 一致，否则 Agent 可能找到语义对象却找不到物理实体。
1. NPC 已有动作、导航和员工 Agent，但动作过渡、复杂社交、多 NPC 对话、失败恢复和长时间稳定性仍是建设重点。
1. 视觉 mesh、碰撞 proxy、交互 site 和实际摆放不一致时，导航、抓取和 handover 可能“看起来正确但执行失败”。
1. LLM 凭据应放在被忽略的 `office_llm.local.json`；SMPL-X 参数、checkpoint、数据集、录屏和生成输出不要提交。示例配置只作为格式模板使用。

## 十四、推荐上手顺序

1. 先运行 `examples/move_joints.py` 或 `examples/office_scene.py --headless`；
1. 阅读 `stretch_mujoco_simulator.py`，理解命令、状态和生命周期；
1. 阅读 `mujoco_server.py`，理解进程、代理和物理循环；
1. 阅读 `robots/base.py` 和 `robots/stretch3/`；
1. 用 `tests/test_robot_interfaces.py`、`test_office_scene.py`、`test_office_semantics.py` 熟悉行为入口；
1. 再按任务进入 `navigations/`、`graspgen/`、`humanoid/`、`agents/` 或 `semantics/`；
1. 最后阅读 `strech_codex/`，理解 MCP 工具编排。

修改代码时先跑对应 focused test，再跑完整测试；如果修改 XML、材质、相机、NPC 或场景布局，还应进行 headless 编译和人工可视化检查。

## 十五、参考文档

- [项目原始 README](../../README.md)：安装和基础示例；
- [MuJoCo Simulator 使用说明](../../docs/using_mujoco_simulator_with_stretch.md)：控制流、状态、相机、传感器和进程模型；
- [Office Semantic World](../../docs/office_semantics.md)：语义关系和 NPC 导航网格；
- [Office Employee Agents](../../docs/office_agents.md)：员工 Agent 和机器人交接；
- [Low-Compute Autonomous Behavior](../../docs/low_compute_behavior.md)：低算力自主行为和 LLM 边界；
- [贡献指南](../../docs/contributing.md)：代码风格和协作；
- [发布到 PyPI](../../docs/releasing_to_pypi.md)：打包和发布。
