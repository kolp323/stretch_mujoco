# NPC 系统指南

本指南是本仓库 NPC（Non-Player Character）系统的当前使用入口。它说明代码和
资产各自的职责、从配置到 MuJoCo 场景的构建流程，以及 Agent、动作回执和会话
如何组合。实现迁移的逐项历史证据在 `aaa_workspace/docs/current.md`；早期设计
目标保留在 `docs/npc_system_optimization.md`，不应把它当作当前接口说明。

## 1. 能力边界

系统将“谁要做什么”和“MuJoCo 中是否已经完成”分开处理。

- `agents/` 维护员工的意图、日程、任务、会话和语义提交；它不会自行宣布动作
  成功。
- `npc/` 为每个 NPC 建立独立的运动、动画、物体附着和命令状态机。
- MuJoCo 是姿态、碰撞、动画帧可见性和 attachment 的事实来源。
- `semantics/` 只在收到成功的物理回执并完成验证后接收世界状态效果。

因此，逻辑任务完成、动画播放完成和物理动作完成不是同一件事。带 marker 的
动作（例如拾取、放置、交接）必须等到对应 marker 和物理回执；移动必须由
`NpcController` 的实时导航完成；未知 NPC、过期 sequence、忙碌状态、缺少 clip
或无效资产都会得到失败回执，而不是静默回退。

当前系统支持 deterministic OBJ mesh-sequence 动画、多人独立控制、导航/转向、
坐下与起立、受控的拾取/放置/交接、可选的 Agent 会话和离线回放。生产 SMPL-X
资产、AMASS 动作和外部纹理是本地受限资产；仓库自带 CesiumMan preview bundle，
仅用于可复现的示例和测试。

## 2. 目录与所有权

```text
stretch_mujoco/
├── npc/                         # embodied NPC 协议、控制器、场景构建
│   ├── protocol.py              # 命令、回执、runtime state 的可序列化契约
│   ├── system.py                # 多 NPC 注册、顺序、幂等和 attachment claim
│   ├── controller.py            # 单 NPC 生命周期与 MuJoCo 交互
│   ├── locomotion.py            # 实时导航和朝向对齐
│   ├── animation/               # graph、marker、mesh-sequence backend
│   ├── appearance_pipeline/     # 外观 catalog、图层和烘焙
│   └── scene_builder.py         # population + manifest -> MJCF
├── agents/                      # 员工意图、动作 recipe、driver、会话
├── semantics/                   # 语义世界；只消费已验证的效果
├── humanoid/                    # SMPL-X/AMASS intake 与 animation baker
├── models/
│   ├── office_population*.json  # population 配置
│   ├── npc_assets.example.json  # 可再分发 preview manifest
│   ├── appearance_recipes/      # 版本化人设/外观 recipe 和 lock
│   ├── accessories/             # 版本化配饰 recipe/runtime 配置
│   └── assets/humanoid/         # preview、sources、private、generated 资产边界
├── tools/                       # build、validate、render 和 intake CLI
└── tests/                       # schema、assets、controller、agent 回归测试
```

运行时的单向数据流如下：

```text
population JSON + asset manifest + scene/profile
                    │  (严格 hash/schema preflight)
                    ▼
            scene_builder -> MJCF -> NpcSystem
                                      │
Agent ActionCommand -> ActionDriver -> NpcCommand -> NpcController -> MuJoCo
                                      │                              │
                                      └──── NpcCommandReceipt ◄───────┘
                                                     │
                                                     ▼
                                           validated semantic effect
```

不要绕过 `NpcSystem.submit()` 直接修改 controller 或语义世界。`NpcSystem` 负责
command ID 幂等、每个 NPC 的递增 sequence、附件的跨 NPC claim 和待消费 receipts；
绕过它会破坏这些保证。

## 3. 快速验证与构建

以下命令在仓库根目录执行。preview population 不需要私有 SMPL-X 资源；production
population 需要本地 `generated/` 和 `private/` 投影已就绪。

```bash
# 1. 验证 population、manifest、散列、OBJ 拓扑和 texture 可读性。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run validate_npc_assets \
  --population stretch_mujoco/models/office_population.json

# 2. 兼容入口：生成可 include 的 NPC-only MJCF。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run build_npc_scene \
  --population stretch_mujoco/models/office_population.json \
  --output /tmp/npc_preview.xml

# 3. 正式入口：由 population.scene 自动组合完整办公室 + NPC MJCF，
#    并写入同名 composition receipt；不接受独立 scene 参数。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run compose_npc_scene \
  --population stretch_mujoco/models/office_population.json \
  --output outputs/npc_preview_office.xml

# 4. 运行核心回归测试。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_assets.py tests/test_npc_schema.py \
  tests/test_npc_system.py tests/test_npc_scene_builder.py
```

标准运行入口也可直接接收 population；它会在系统临时目录生成/复用组合 MJCF，且
`population.scene` 是唯一基础场景来源。可选 `--semantics` 用于新场景的显式语义图：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run launch_sim --headless \
  --population stretch_mujoco/models/office_population.json \
  --semantics stretch_mujoco/models/office_semantics.json
```

schema-v2 的 `population.trajectory_profile` 是路线事实源，会直接传入 `NpcSystem` 和
action driver；MJCF 的 `npc_trajectory_profile` custom text 只供 schema-v1/legacy scene
fallback，新 population 不需要同时写它。旧 schema-v2 文件中的 `agent_id` 仍可读取，
但它只是兼容别名；具身运行时、语义对象和动作协议统一以 `npcs` 的映射键 `npc_id` 为准。
`capabilities` 会在 planner 和 runtime action validation 中约束 locomotion、sit、
conversation、object handover 与 computer use；idle/stand-up 保持 recovery 可用。
新语义场景可在 interaction point 的 attributes 中用 `binding` 声明 `location`、
`seat_navigation`、`placement`、`object_approach` 或 `robot_request`；对话/交接 role site
使用 `conversation_role`/`handover_role` 加 `participants` 与 `participant`。可选 `yaw`
随 site 一起声明，避免在 Python 中加入场景专用坐标或名称。

schema-v2 population 可用 `interaction_templates` 为整个 `npcs` roster 声明 wildcard
互动站位。`conversation` 必须有 `speaker` 和 `listener`，`handover` 必须有 `giver` 和
`receiver`；每个角色写 `{ "site": "...", "yaw": ... }`，其中 `yaw` 可省略。模板只会
为 roster 中不同的两个 NPC 展开：每个有序 pair 都有一组角色站位，所以 `(A, B)` 的第一位
分别是 speaker/giver，`(B, A)` 则交换实际扮演该角色的 NPC。模板 site 必须是该 scene 的
真实 site；运行时以 semantic world 的 interaction points 校验，composition 也会对编译后的
MuJoCo model 复核。

站位优先级是显式 semantic `conversation_role`/`handover_role` pair binding、population
wildcard template、旧的 legacy fallback，依次降低。显式 handover binding 的 yaw 也按角色
逐项覆盖：未声明的角色继续继承 wildcard yaw；没有模板时才使用 legacy yaw。迁移旧
schema-v2 population 时，在保留 `schema_version: 2` 的前提下加入模板，并把原先 Python
常量对应的 site 名称写入配置；不含模板的旧 schema-v2 与 schema-v1 会继续使用 legacy
fallback。

生产 population 的常用预检是：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run validate_npc_assets \
  --population stretch_mujoco/models/office_population.production.example.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run build_npc_scene \
  --population stretch_mujoco/models/office_population.production.example.json \
  --output /tmp/npc_production.xml
```

`build_npc_scene` 在写入前加载基场景、验证 population 与 manifest，并为每个 NPC
生成稳定命名的 mocap body、visual mesh frame、碰撞体和 interaction site。它保留为
NPC-only include 的兼容入口；追加 `--include-base-scene` 可以生成 combined MJCF。

正式加载使用 `compose_npc_scene`：它以 `population.scene` 为唯一基场景来源，不允许
再传一个会冲突的 scene 参数；产物编译 MuJoCo、验证每个 canonical body、handover
site 与 spawn site，并写入 population、base scene、manifest 和生成场景的 SHA-256
receipt。相同输入会复用 receipt 完整且可编译的产物。`load_composed_npc_runtime()` 是
Python facade，会从同一个 population 同时构造 `MjModel`、`NpcSystem`、
`SemanticWorld` 与 `OfficeAgentRuntime`，避免 body 和 agent ID 分别加载导致的
`unknown_npc`/semantic binding conflict。

MuJoCo 的 `MjModel` 编译后不能热插入 body；要改变人数、appearance 或 spawn，请改
population 后重新 compose/reload。`recording/native_scene.py` 的 humanoid clone 仅是
legacy recording fixture，不能作为 production NPC 加载路径。生成的 MJCF、wrapper、
receipt 与视频都应写入被忽略的 `outputs/` 或 `/tmp`，不得加入版本控制；基础 office
XML 不会被该流程修改。

## 4. Population 配置

`models/office_population.json` 是可分发 preview 示例，
`models/office_population.production.example.json` 是需要本地受限资产的 production
示例。schema 当前为 v2。顶层关键字段如下：

```json
{
  "schema_version": 2,
  "scene": "office_scene.xml",
  "asset_manifest": "npc_assets.example.json",
  "clock": {"start": "09:00", "minutes_per_second": 1.0},
  "npcs": {
    "employee_01": {
      "profile": {"display_name": "Alex Chen", "role": "Operations Specialist"},
      "embodiment": {
        "bundle": "cesium_man_preview_v1",
        "appearance": "cesium_default_v1",
        "animation_graph": "office_humanoid_v1",
        "collision_profile": "adult_humanoid_v1",
        "scale": 1.0
      },
      "spawn": {"location": "workstation_left", "site": "desk_left_work_site", "yaw": 3.1415926},
      "capabilities": ["locomotion", "sit"],
      "needs": {"hunger": 0.22, "thirst": 0.30, "fatigue": 0.18},
      "schedule": []
    }
  }
}
```

`profile` 由 Agent 层使用；`embodiment` 选择经 manifest 认证的 bundle、appearance、
动画图和尺度；`spawn.site` 必须存在于场景；capabilities 决定 Agent 可规划的行为。
当顶层提供 `appearance_catalog` 时，每个 NPC 还必须提供 `visual_identity`，并且
`appearance_config` 中的 `skin`、`hair`、`top`、`bottom`、`shoes` 选择必须与该
identity 的 catalog 图层完全一致。

新增 production NPC 的最小流程：先登记/烘焙 appearance 和 bundle，再在 population
里引用它，最后执行 asset preflight 和 scene build。不要只编辑 XML 来复制一个人形；
那会绕过 manifest、hash、spawn 和 Agent-ID 校验。

## 5. Asset manifest 与动画图

`NpcAssetManifest`（当前 schema v1）是运行时资产的入口。一个 bundle 要声明：

- 坐标与尺寸：`format`、`topology_id`、`coordinate_system: "mujoco_z_up"`、
  `unit: "meter"`、`height_m`；
- 材质：不重复的 `material_slots`，以及每个 appearance 的 texture 路径与同一
  topology ID；
- 动画：每个 clip 的 `fps`、`loop`、`root_motion`、OBJ `frames` 和可选 marker；
- 配饰：每个 accessory 的 mesh 和逐帧 anchor；
- 完整性：所有被引用文件的 `sha256` 和 `asset_quality`。

`preview` bundle 至少需要 `idle`；`production` 与 `restricted` bundle 必须满足
`OFFICE_CLIPS` 的完整 clip 合约。预检还会检查 hash、PNG 解码、OBJ 顶点/UV/面拓扑
的一致性，以及 Cesium preview 不能伪装成 SMPL-X production。manifest 相对路径一律
相对于 manifest 自身，不要依赖当前工作目录。

动画图从已验证的 manifest 生成，而不是由 driver 内嵌一个不受资产约束的 clip
列表。marker 是物理语义的完成条件，例如 `grasp`、`release`、`standing`；若
requested clip 不存在，controller 会发出 fallback/failure，而不是把动作视为成功。

## 6. 外观、人设与配饰

`models/appearance_recipes/office_personas_v1.roster.json` 是 production 人设的
版本化输入。它引用 flat、hair、face-detail 和 textile layer spec；
`tools/build_npc_persona_roster.py` 生成 catalog、atlas、thumbnail、appearance receipt
并更新本地生成 manifest。该工具可以重建派生投影，不能接受未经审查的 source。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/build_npc_persona_roster.py \
  --roster stretch_mujoco/models/appearance_recipes/office_personas_v1.roster.json \
  --asset-root stretch_mujoco/models/assets/humanoid/generated/animations \
  --manifest stretch_mujoco/models/assets/humanoid/generated/animations/manifest.json
```

纺织 layer 必须在版本化 spec 中钉住 archive SHA-256、source URL、license、archive
member、semantic mask、输出路径和 seed。来源 ZIP 存于 `assets/humanoid/sources/`，
生成的 RGBA layer、receipt 和 thumbnail 存于 `generated/`。生成命令拒绝 hash 不符
的 archive 和越出 asset root 的输出路径。

OBJ/GLB 配饰以 `models/accessories/<accessory_id>.recipe.json` 和
`<accessory_id>.runtime.json` 为唯一版本化事实源。源文件在 `sources/`，规范化 OBJ、
anchor、fused frame 和 runtime manifest 在 `generated/`。修改配饰 pose 后必须从 recipe
重新构建，不能手改某一帧的 generated OBJ。

资产目录的管理规则为：

| 资产类型 | 位置 | 可提交性 |
| --- | --- | --- |
| 可再分发 preview fixture | `models/assets/humanoid/` | 提交，并在 manifest 中固定 hash |
| 配置、recipe、manifest 模板、selection example | `models/` | 提交 |
| 审核后的 source archive | `models/assets/humanoid/sources/` | 本地共享，不提交 payload |
| SMPL-X/AMASS 和受限 receipt | `models/assets/humanoid/private/` | 本地，不提交 |
| OBJ frame、atlas、anchor、生成 manifest、视频 | `models/assets/humanoid/generated/` 或 `/tmp` | 本地，不提交 |

`sources/`、`private/`、`generated/` 是主工作区共享资源树；其他维护 worktree 应链接到
它们的物理所有者，不能保留分支私有副本。详情见各目录 README 和仓库 `AGENTS.md`。

## 7. 运行时命令、回执与状态

公开协议定义在 `stretch_mujoco.npc.protocol`：

- `NpcCommand`：不可为空的 command ID、每 NPC 单调递增的 `sequence`、目标 NPC、
  kind、payload、发出时间和可选 deadline；
- `NpcCommandKind`：`move_to`、`align_to`、`play_animation`、`attach_object`、
  `detach_object`、`interaction_cue`、`cancel`；
- `NpcCommandReceipt`：`accepted`、`running`、`succeeded`、`failed`、`cancelled` 或
  `timed_out`；后四种是终态；
- `NpcRuntimeState`：实体位置/四元数、locomotion、requested/resolved clip、phase、
  marker、route、held objects、active command 与最后 receipt 的只读快照。

最小的原生集成循环如下。实际服务端已经将这一步封装；示例用于说明 ownership。

```python
import mujoco
from stretch_mujoco.npc import NpcCommand, NpcCommandKind
from stretch_mujoco.npc.system import NpcSystem

model = mujoco.MjModel.from_xml_path("/tmp/npc_preview.xml")
data = mujoco.MjData(model)
system = NpcSystem.from_model(model, scene_path="stretch_mujoco/models/office_scene.xml")

receipt = system.submit(NpcCommand(
    command_id="employee_01-move-0001",
    sequence=0,
    npc_id="employee_01",
    kind=NpcCommandKind.MOVE_TO,
    payload={"site": "desk_right_work_site", "speed": 0.6},
    issued_at=0.0,
    deadline=30.0,
))
assert receipt.status.value == "accepted"

while not any(item.status.terminal for item in system.drain_receipts()):
    mujoco.mj_step(model, data)
    system.step(model, data, data.time)
```

真实服务端应在每个物理 step 调用 `system.step(model, data, sim_time)`，再用
`drain_receipts()` 把新 receipt 交给 Agent driver。不要用“提交成功”取代最终 receipt；
`accepted` 只说明 controller 接受了命令。

## 8. Agent driver、交接与会话

`MujocoNpcActionDriver` 将 `ActionCommand` 降级为一个或多个 `NpcCommand`，并只在
全部相关 receipt 成功后返回 `ExecutionStatus.SUCCEEDED`。它维护 per-NPC sequence、
action deadline、移动后 location、object workflow 和 handover barrier。构造标准 office
driver 时使用：

```python
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver

driver = create_mujoco_action_driver(
    simulator,
    npc_ids=population.npcs,
    interaction_templates=population.interaction_templates,
)
```

其中 `simulator` 必须实现 `pull_status()`、`submit_npc_command()`、
`pull_npc_receipts()` 和 `cancel_npc_command()`。生产动作的可用集合会被 manifest
中已批准的 clips 收窄；缺 clip 的 action 在发出物理命令前失败。

### 工位 work 会话

提交 `ActionType.WORK` 不会让 NPC 在工位旁站立工作。runtime 会将其展开为
`MOVE_TO(chair) → SIT(chair) → WORK(workstation) → STAND_UP(chair) → IDLE`，并等待每一步
的终态回执；特别是 `WORK` 只有在 `SIT` 已提交 chair occupancy 后才会开始，椅子仅在
`STAND_UP` 成功后释放。

场景通过语义关系而非 object ID 命名定义这条规则：每个可工作的 workstation 必须有且仅有
一条 `Chair --NEAR--> Workstation`。用于实体执行时，该 chair 必须有一个带有限
`attributes.yaw` 的 `chair_sit_site`，workstation 必须有一个 `desk_work_site`。缺少或歧义
任一项会拒绝 action/driver 装配，避免场景迁移后静默退化为站立 work。

`agents/conversation.py` 的 `ConversationSession` 归 runtime 所有，记录 participants、
topic、turn、transcript、timeout、status 与 interrupt policy。逻辑/mock 会话和实体接入
是不同层：只有 `InteractionDriver` 在每个参与者的接近、朝向和 talk cue receipt
均成功后，才可以把会话推进为实体可播放的 turn。LLM（如启用）仅提供受 schema
约束的候选文本/意图；会话状态、超时、中断、冷却、审计事件和最终效果仍由确定性
runtime 验证。

可生成无 LLM 的对话验收回放：

```bash
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python \
  examples/npc_conversation_acceptance.py \
  --output-dir /tmp/npc-conversation --render both --fps 5 --width 640 --height 360
```

该回放是 session/receipt 的验证证据；它不会替代 production 资产验证，也不能证明
尚未实际执行的物理行为。

## 9. 场景与轨迹配置

schema-v2 population 应通过 `trajectory_profile` 声明路线配置；只有 legacy scene
直接由 `NpcSystem.from_model()` 加载时，才读取 MuJoCo custom text 中的
`npc_trajectory_profile`。
`npc/trajectory_profiles/office_v1.json` 将语义 anchor（site + role）和允许的 route
分开，并钉住源 MJCF 的 SHA-256。运行时从实时碰撞几何重新规划，profile 不保存容易
过期的世界坐标 waypoint。

新增场景时，请按以下顺序操作：

1. 在 MJCF 中定义稳定的 NPC spawn、interaction 和 navigation site。
2. 新建版本化 trajectory profile，声明 anchors、routes、允许 action 和 scene hash。
3. 运行路径预检；失败时修改场景/profile，而不是把硬编码坐标写进 Agent。
4. 在 population 中引用新 scene、site 和 profile，再运行 asset/scene 验证。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/validate_npc_trajectories.py \
  --scene stretch_mujoco/models/office_scene.xml \
  --profile stretch_mujoco/npc/trajectory_profiles/office_v1.json \
  --receipt /tmp/office_v1_trajectory_receipt.json
```

receipt 是一次运行的审计投影，不是后续运行时配置。若场景 bytes、site、profile hash
或 route 可达性漂移，preflight 必须失败。

## 10. 修改检查清单

更改配置、动作或资产后，至少执行与改动相符的检查：

```bash
# Python contract / asset / scene regression
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_assets.py tests/test_npc_schema.py tests/test_npc_system.py \
  tests/test_npc_scene_builder.py tests/test_animation_controller.py

# Production payload 已配置时的严格 preflight
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run validate_npc_assets \
  --population stretch_mujoco/models/office_population.production.example.json
```

当变更外观、人设或配饰时，还要验证 roster/catalog、所有被引用的 SHA-256、manifest
和 production population 的同步；当变更 scene/profile 时，再运行 trajectory
preflight。对生产外观需要使用原办公室 MJCF 的 native lighting 生成验收 render；不要
把 review-only 灯光或替代场景伪装为真实验收。

## 11. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| `asset is missing` 或 hash mismatch | manifest 相对路径、文件散列、`sources/private/generated` 的共享投影；不要修改 hash 来掩盖文件漂移。 |
| `restricted production clips are incomplete` | 完整批准的 `OFFICE_CLIPS`、每个 OBJ frame 和 manifest hash；不能用 preview frame 补缺。 |
| `unknown_npc` / `unknown_target_site` | population ID、构建后的 MJCF 命名和对应 scene site。 |
| `stale_sequence` / `npc_busy` | 同一 NPC 的 driver sequence 和未完成命令；等待 terminal receipt 或先 cancel。 |
| action 一直不完成 | deadline、marker 名称、route/site 和 MuJoCo 的 controller step 是否每帧调用。 |
| 会话没有实体动作 | 区分 logical session 与 embodied path；检查所有参与者的 movement/alignment/talk receipt，而不是仅检查 transcript。 |

如果完整 production 资产不可用，使用 `office_population.json` 运行 preview 验证，并在交付
中明确标记为 preview；不要把缺失或不受限的资源改名为 production。
