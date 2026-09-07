# NPC 系统改进实施说明

## 1. 简要说明

原项目实际上有两套互不统一的 NPC 表现路径：在线仿真只能通过一组全局字段控制固定名为 `humanoid_preview` 的单个角色；离线录制则使用 `em01_*`、`em02_*` 等另一套命名和切帧逻辑。与此同时，Agent 的 `MOVE_TO` 等动作只要逻辑计时结束就会直接修改语义状态，并不等待 MuJoCo 中的角色真正到达目标。

本次改动增加了一条统一的 NPC 主链：

```text
Agent 动作意图
    -> ActionDriver
    -> 有序 NpcCommand 队列
    -> NpcSystem
    -> 独立 NpcController
    -> MuJoCo 位姿/动画
    -> NpcCommandReceipt
    -> 验证后提交 SemanticWorld
```

其核心意义是把“计划做什么”“MuJoCo 实际做到了什么”和“语义世界最终承认什么”分开。这样，多 NPC、失败、超时、取消和回放都可以沿同一条链路继续扩展，而不再依赖更多全局字段或分散的 `if/elif`。

## 2. 原项目的主要限制

### 2.1 在线控制只有一个全局 NPC

原来的 `MujocoServerProxies` 只有以下全局状态：

- humanoid animation；
- sit target；
- navigation target；
- playback speed。

`MujocoServer` 也只创建一个 `HumanoidMeshAnimator`。动画器内部固定查找 `humanoid_preview` body 和 `humanoid_preview_frame_*` geom，并独占一份导航、落座、当前帧和 root pose 状态。因此，第二个 NPC 无法拥有独立的动画或运动生命周期。

### 2.2 在线与离线使用不同的命名和动画规则

在线动画器识别 `humanoid_preview_frame_*`，而离线 renderer 识别 `em01_frame_*`、`em02_frame_*`。两处分别维护正则、clip fallback 和帧选择规则，容易产生表现差异。

### 2.3 逻辑完成不代表物理完成

原来的 `OfficeAgentRuntime` 使用固定逻辑时长执行动作。以 `MOVE_TO` 为例，倒计时结束后便直接更新 `EmployeeState.location`，即使 MuJoCo 进程退出、导航失败或 NPC 根本没有到达目标，逻辑位置仍可能被错误提交。

### 2.4 资产缺少可验证身份

原项目主要依靠 XML 中约定的 OBJ、PNG 路径，没有统一 manifest 来证明：

- bundle 和 appearance 是否存在；
- clip 是否有帧；
- OBJ 拓扑和 UV 是否一致；
- texture 是否属于同一 topology；
- 文件内容是否与预期 SHA-256 一致；
- CesiumMan preview 是否被误当成正式 SMPL-X 资产。

## 3. 本次新增的模块和职责

本次新增 `stretch_mujoco/npc/` 包，作为 NPC 物理运行链路的唯一所有者。

### 3.0 外观烘焙流水线（后续增量）

文件：

- `stretch_mujoco/npc/appearance_pipeline/bake.py`
- `stretch_mujoco/npc/appearance_pipeline/slots.py`

新增的烘焙器以 JSON recipe 为唯一事实源：每个 material slot 指向一张 base PNG，并可按顺序叠加带 alpha 的图层、可选灰度/alpha mask、`normal` 或 `multiply` blend 和 opacity。它输出确定性的 PNG、SHA-256、recipe SHA-256 及 `*.appearance.json` sidecar。可选注册步骤会在通过既有 `NpcAssetManifest` 校验后，原子写入目标 bundle 的 `appearances` 和 `sha256`；因此生成纹理不会成为未登记或拓扑不匹配的孤儿资产。

```bash
uv run bake_npc_appearance \
  --recipe assets/looks/employee_blue.recipe.json \
  --output-dir stretch_mujoco/models/assets/humanoid/generated/employee_blue_v1 \
  --asset-manifest stretch_mujoco/models/npc_assets.production.json \
  --bundle smplx_neutral_v1
```

recipe 的最小形式如下（路径相对 recipe 文件）：

```json
{
  "schema_version": 1,
  "appearance_id": "employee_blue_v1",
  "texture_topology_id": "smplx-neutral-10475-v1",
  "textures": {
    "body": {
      "base": "base_skin.png",
      "layers": [
        {"image": "blue_shirt.png", "mask": "torso_mask.png"}
      ]
    }
  }
}
```

`slots.py` 只保留 `body/hair/top/bottom/shoes` 的命名占位和明确失败接口，尚不生成 mesh 或 geom。这是刻意限制：当前的单人体 mesh 若复制成多个重叠 geom 再分别赋材质，会产生重叠渲染及不正确的碰撞。只有正式资产已经按这些区域权威拆分为独立 mesh 后，才应扩展该模块来装配分槽几何；在此之前一律使用烘焙纹理。

对于当前由 `_write_material_texture()` 生成的 SMPL-X 平色 atlas，另有 `flat_layers.py`。它只识别生成器定义的 skin/shirt/pants/shoes RGB 区域，并生成透明、UV 对齐的编辑层；它不是通用的人体语义分割器。示例 `alex_warm_v1` 的 layer spec 和 recipe 位于 `assets/humanoid/generated/animations/appearance_sources/alex_warm_v1/`，可依序运行：

```bash
uv run make_npc_flat_layers --spec .../alex_warm_v1/layers.json --output-dir .../alex_warm_v1/layers
uv run bake_npc_appearance --recipe .../alex_warm_v1/recipe.json \
  --output-dir .../appearances/alex_warm_v1 \
  --asset-manifest .../animations/manifest.json --bundle smplx_office_neutral_v1
```

### 3.1 Population schema

文件：`stretch_mujoco/npc/schema.py`

新增的主要模型包括：

- `NpcPopulation`：场景、资产 manifest、时钟和 NPC 定义集合；
- `NpcDefinition`：一个 NPC 的 profile、embodiment、spawn、capabilities、needs 和 schedule；
- `NpcEmbodiment`：bundle、appearance、animation graph、collision profile 和 scale；
- `NpcSpawn`：语义位置、MuJoCo site 和初始 yaw。

配置版本明确固定为 schema v2。解析时会拒绝：

- 未知 schema version；
- 缺失必填字段；
- 重复 JSON key；
- 非法 scale；
- 未注册 capability；
- 非法 schedule 结构；
- 调用方提供语义集合时，不存在的 location 或 site。

`EmployeeAgent.from_definition()` 接受已经校验过的 `NpcDefinition`。旧的 schema v1 `employees` 配置仍由原来的 `from_dict()` 读取，因此旧 demo 和测试无需立即迁移。

### 3.2 资产 manifest 与校验

文件：`stretch_mujoco/npc/assets.py`

新增 `NpcAssetManifest`、`AssetBundle`、`AppearanceManifest` 和 `ClipManifest`。启动前校验包括：

- mandatory `idle` clip；
- bundle、appearance 和 population 引用；
- material slot 合法性；
- topology ID 与 texture topology ID 一致；
- OBJ 顶点数量、face index 和 UV index 拓扑一致；
- PNG 能否解码及通道数量；
- 所有引用文件均有 SHA-256，且实际内容匹配；
- 坐标系为 `mujoco_z_up`、单位为 meter；
- CesiumMan 资源只能显式声明为 preview，不能伪装成 SMPL-X bundle。

新增示例 `stretch_mujoco/models/npc_assets.example.json`。它明确使用 `cesium_man_preview_v1` 和 `asset_quality="preview"`，不会静默冒充生产级 SMPL-X 资产。

### 3.3 统一命名和 MuJoCo binding

文件：

- `stretch_mujoco/npc/naming.py`
- `stretch_mujoco/npc/binding.py`

新生成对象使用统一命名，例如：

```text
npc__employee_01
npc__employee_01__clip__walk__frame__000__slot__body
npc__employee_01__collision__torso
npc__employee_01__handover
```

业务代码不再自行拼接这些字符串。`NpcBinding.from_model()` 在 MuJoCo model 编译后一次性解析 body、mocap、frame geom、interaction site 和 collision geom ID。

为了保持兼容，统一解析器目前同时识别：

- 新的 canonical `npc__*` 命名；
- 旧单 NPC `humanoid_preview_frame_*`；
- 旧多 NPC `em01_frame_*`、`em02_frame_*`。

在线系统和离线 renderer 都复用这个解析器，删除了第二份命名规则事实源。

### 3.4 命令、回执和状态协议

文件：`stretch_mujoco/npc/protocol.py`

新增：

- `NpcCommand`；
- `NpcCommandKind`；
- `NpcCommandReceipt`；
- `CommandStatus`；
- `NpcRuntimeState`。

命令包含 `command_id`、`sequence`、`npc_id`、kind、payload、issued time 和可选 deadline。`command_id` 用于幂等，`sequence` 用于拒绝旧命令。

状态不再只有一个 animation 字符串，而是包含：

- MuJoCo position 和 quaternion；
- locomotion 状态；
- requested animation 与实际 resolved clip；
- clip phase；
- active command；
- 最近一次 receipt；
- 单调递增 revision。

当前协议已经声明并实现 move、align、play animation、attach、detach、interaction cue 和 cancel。附件命令只接受带 free joint 的 MuJoCo body；interaction cue 携带 interaction ID，并通过与普通动画相同的回执链报告完成或失败。

### 3.5 每 NPC 独立 Controller

文件：

- `stretch_mujoco/npc/system.py`
- `stretch_mujoco/npc/controller.py`
- `stretch_mujoco/npc/locomotion.py`
- `stretch_mujoco/npc/animation/`

`NpcSystem` 只负责发现、注册、路由、幂等和状态汇总。每个 `npc_id` 都有一个独立 `NpcController`，分别拥有：

- mocap root；
- locomotion target 和时间状态；
- animation requested/resolved clip 和 phase；
- active command；
- last receipt 和 revision。

`MeshSequenceBackend` 封装了现有 OBJ alpha 切帧方式。每个 backend 只修改自己 binding 中的 geom，因此控制 employee_01 不会改变 employee_02 的透明度或内部帧状态。

世界平移和 yaw 只由 `LocomotionController` 写入 mocap root；动画 backend 只负责视觉采样。这为以后替换 articulated/skinned backend 保留了稳定边界。

### 3.6 场景生成

文件：`stretch_mujoco/npc/scene_builder.py`

新增 `build_npc_scene()`：

1. 读取并校验 population；
1. 读取并校验 asset manifest；
1. 从源场景解析 spawn site 的世界位置；
1. 为每个 NPC 生成 canonical mocap body、frame geom、collision geom 和 handover site；
1. 相同 bundle、scale、clip 和 frame 共享 mesh asset；
1. 在输出中记录 population 与 manifest 的 source hash；
1. 输出可以独立被 MuJoCo 编译、也可以作为 include 被场景使用的 MJCF。

新增 `stretch_mujoco/models/office_population.json`，定义两个显式使用 preview bundle 的 NPC。当前公开 preview manifest 只有 idle clip；完整 idle/walk/sit/work/eat 序列仍可由现有 office scene 和旧多 NPC 场景通过兼容 binding 使用。后续正式资产应通过同一 manifest 增加 clip，而不是在业务代码中添加特殊分支。

新增命令：

```bash
uv run validate_npc_assets --population stretch_mujoco/models/office_population.json
uv run build_npc_scene \
  --population stretch_mujoco/models/office_population.json \
  --output /tmp/stretch_npc_scene.xml
```

### 3.7 跨进程服务器集成

修改文件：

- `stretch_mujoco/mujoco_server.py`
- `stretch_mujoco/stretch_mujoco_simulator.py`

`MujocoServerProxies` 新增：

- 有序 command queue；
- 按 NPC 发布的 state dict；
- receipt queue。

这里使用 queue，而不是“每个 NPC 只保存最新命令”的 dict。原因是连续命令不能被后一个覆盖，否则 attach/detach、取消或同步 cue 会丢失。

服务器每个 control callback 的顺序现在是：

1. 消费全部待处理命令；
1. 调用一次 `NpcSystem.step()`；
1. 发布所有 NPC observed states；
1. 发布新增 receipts；
1. 继续机器人附件、物体可见性和语义快照逻辑。

模拟器新增公开 API：

```python
submit_npc_command(command) -> str
cancel_npc_command(npc_id, command_id) -> None
pull_npc_states() -> dict[str, NpcRuntimeState]
pull_npc_receipts() -> tuple[NpcCommandReceipt, ...]
```

旧的 `set_humanoid_*` API 仍然存在，但 `set_humanoid_animation()` 会发出 deprecation warning，并转发到默认 `employee_01` 的新命令链路。它不再是新代码推荐的入口。

服务器关闭时，仍在运行的 NPC 命令会产生 `FAILED(reason="simulator_restarted")`，让上层决定重试或补偿，而不是无限停留在 running。

### 3.8 Agent 物理执行桥

文件：

- `stretch_mujoco/agents/drivers.py`
- `stretch_mujoco/agents/action_recipes.py`
- `stretch_mujoco/agents/simulation_bridge.py`

`ActionExecution` 新增 execution ID、phase、started time、deadline 和 driver handle。`MujocoNpcActionDriver` 首轮接管 `MOVE_TO`，后续又接管了 `SIT`、`PICK_UP`、`PUT_DOWN` 和 `HANDOVER`：

1. 将语义 location 映射到 MuJoCo site；
1. 提交带 deadline 的 `NpcCommand.MOVE_TO`；
1. 轮询 receipt；
1. 只有 `SUCCEEDED` 才让 runtime 执行 semantic commit；
1. failed、cancelled 或 timed out 均保持原逻辑位置，并进入失败恢复。

未启用 embodied driver 时，runtime 继续使用原来的定时行为，因此纯逻辑单元测试和低算力运行方式保持兼容。创建 runtime 时可通过 `embodied=True` 启用 MuJoCo driver。

语义提交使用 `execution_id` 做进程内幂等记录，同时事件中加入 execution ID，便于把 action、NPC command 和 semantic commit 串联起来。

### 3.9 Snapshot v2 和离线回放

修改文件：

- `stretch_mujoco/recording/snapshots.py`
- `stretch_mujoco/recording/renderers.py`

新 snapshot writer 使用 schema v2，并记录 `npcs` 下的 observed pose、quaternion、logical action、execution ID、locomotion、clip、phase、held objects 和 active command。

当调用方传入 `npc_states` 时，snapshot 使用 MuJoCo observed state；不再重新推断物理位姿。为了兼容当前 2-D 工具，仍输出一个由权威状态派生的 `agents` projection。

v1 JSONL 默认仍可原样读取，避免破坏历史录制；调用 `read_snapshots(..., upgrade_v1=True)` 或 `adapt_snapshot_v1()` 可以得到最小 v2 投影。

离线 3-D renderer 使用 snapshot 中记录的 animation phase 选择帧，不再始终按输出视频的全局 frame index 取模。

原 `.gitignore` 曾将整个 `/stretch_mujoco/recording/` 目录忽略，这会连 Python 源码一起排除。本次将规则收窄为仅忽略该目录中的 JSONL、MP4 和 recording manifest，使录制模块源码可以进入版本控制。

## 4. 关键行为差异

### 4.1 原来的 MOVE_TO

```text
提交 MOVE_TO
    -> 逻辑倒计时
    -> 直接修改 agent.location
```

这个流程无法区分“计划耗时结束”和“物理到达”。

### 4.2 现在启用 embodied driver 后的 MOVE_TO

```text
提交 MOVE_TO
    -> validate
    -> 创建 ActionExecution
    -> 提交 NpcCommand
    -> NPC root 实际移动和对齐
    -> SUCCEEDED receipt
    -> verify
    -> 幂等提交 agent.location
```

失败路径为：

```text
FAILED / CANCELLED / TIMED_OUT / simulator_restarted
    -> 不修改 agent.location
    -> 释放临时预占
    -> 记录 action_failed 与失败原因
    -> agent 回到 available/idle
```

这是本次重构最重要的语义保证。

## 5. 兼容性策略

- schema v1 employee 配置继续支持；schema v2 使用严格 population parser。
- 旧单 NPC 和旧多 NPC geom 名称继续识别。
- 旧 `set_humanoid_*` 方法保留兼容转发。
- 不启用 embodied driver 时继续使用原定时 action 行为。
- snapshot v1 继续可读，v2 writer 不再新增 v1 数据。
- 当前 OBJ mesh sequence 继续使用，没有用 alpha crossfade 冒充骨骼 blend。

这些 adapter 的目的只是让迁移可分阶段进行。新增功能应直接使用 population、`NpcSystem`、`NpcCommand` 和 observed snapshot，不应继续扩大旧接口。

## 6. 测试与验证

新增测试覆盖：

- population schema v2 正常解析；
- 未知版本、非法 scale、未知 capability、非法 location/site 被拒绝；
- preview asset manifest、SHA-256 和 topology 校验；
- scene builder 输出可由 MuJoCo 编译；
- 两个 NPC 同时移动到不同 site；
- 两个 NPC 的 mocap pose、animation geom 和 controller 状态隔离；
- command ID 幂等；
- stale sequence 被拒绝；
- deadline 字段校验、超时 receipt 向 Agent 层传播；
- `MOVE_TO` 成功回执前语义位置不变化；
- 物理失败后语义位置不变化；
- snapshot v1 兼容与 v2 输出。

取消与 `simulator_restarted` 终结逻辑已经实现，但本轮新增测试中尚未对真实跨进程关闭场景做专项集成覆盖。

实际执行结果：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests
205 passed

uv run mypy --ignore-missing-imports stretch_mujoco/npc ...
Success: no issues found in 17 source files

新增核心文件 flake8
通过

validate_npc_assets
通过

生成 MJCF 后调用 mujoco.MjModel.from_xml_path()
成功，识别 2 个独立 mocap NPC
```

需要设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，是因为当前机器的系统 ROS pytest 插件会自动加载，但其环境缺少 `lark`；这不是项目测试失败。

仓库级 `pre-commit --all-files` 仍会被原项目已有的 flake8 和 codespell 基线问题阻断，例如旧文件中的 unused import、超长行、bare except 和拼写问题。本次新增核心文件的格式、flake8 和 mypy 检查均通过，没有通过跳过测试或放宽断言来隐藏失败。

## 7. 首轮结束时未实现的内容

本次完成的是方案中的首轮承重链路，而不是六个阶段的全部功能。以下仍属于后续工作：

- 完整 AnimationGraph、marker 中断和动作过渡；
- stand up、turn、talk、listen、wave、handover 等正式动画资产；
- articulated/skinned animation backend 和真实 pose blend；
- NPC 手部附件控制；
- NPC–NPC、NPC–robot conversation session；
- 经物理验证的 robot/NPC handover；
- reservation 与 interaction coordinator；
- 动态障碍重规划；
- 完整 replay command/event 流；
- 删除旧 `HumanoidMeshAnimator` 和旧 proxy 字段。

上述清单是首轮里程碑结束时的状态。第 9 节记录按照建议顺序继续实施后的最新状态；其中 attachment、interaction、marker 与 handover 主链已经不再属于未实现项。

## 8. 本轮采用的后续实施顺序

建议沿当前主链继续推进：

1. 将正式 idle/walk/sit/work/eat 资产登记进 manifest，并让生产 population 不再依赖旧 XML frame 区域；
1. 为 `NpcController` 增加 sit 和 attachment 完成条件；
1. 实现 animation graph、marker 和 fallback event；
1. 将 `PICK_UP`、`PUT_DOWN` 接入物理 receipt；
1. 实现通用 `InteractionCoordinator`；
1. 先完成 robot→NPC handover，再复用同一定义完成 NPC→NPC handover；
1. 等所有调用方迁移后删除旧全局 humanoid 路径。

这种顺序能保持唯一事实来源：MuJoCo 拥有 observed pose 和 attachment，controller 拥有动画状态，Agent 拥有计划意图，SemanticWorld 只保存已经验证并提交的结果。

## 9. 按建议顺序完成的第二轮改造

### 9.1 正式动画资产登记

修改文件：

- `stretch_mujoco/humanoid/smplx_animation_baker.py`
- `stretch_mujoco/models/office_population.production.example.json`
- `stretch_mujoco/models/assets/humanoid/generated/README.md`
- `stretch_mujoco/npc/assets.py`

原烘焙器虽然会生成 idle、walk、sit、work、eat OBJ，但输出的是旧的内部帧清单，不能被 `NpcAssetManifest` 或 `build_npc_scene` 直接消费。本轮将烘焙结果改成正式 manifest：

- bundle 固定为 `smplx_office_neutral_v1`；
- 显式声明 production quality、SMPL-X topology、米制和 MuJoCo Z-up；
- 登记 idle、walk、sit、work、eat 的 fps、loop、root motion 和全部帧；
- 登记 `left_foot`、`right_foot`、`seated`、`work_cycle`、`consume` marker；
- 自动计算所有 OBJ 和 PNG 的 SHA-256；
- production bundle 缺少上述五个基础 clip 时直接校验失败。

新增 `write_npc_asset_manifest()`，因此已有本地烘焙帧无需重新加载 SMPL-X 模型也能重建 manifest。当前机器上的受限资产已据此生成并通过校验，但 `generated/` 仍按许可证要求被 Git 忽略。

`office_population.production.example.json` 是生产配置入口，它直接引用本地生成的 manifest，不再引用旧 XML 中手写的 frame 区域。公开的 `office_population.json` 仍保留 CesiumMan preview，因为仓库不能重新分发受限 SMPL-X 参数或派生人体网格。这里保留两条显式路径，比在缺少私有资产时静默降级更安全。

### 9.2 动画图、marker 和 fallback

修改文件：

- `stretch_mujoco/npc/animation/controller.py`
- `stretch_mujoco/npc/controller.py`
- `stretch_mujoco/npc/protocol.py`

`AnimationController` 现在维护 clip definition、loop/non-loop 行为、transition 和 marker 穿越事件。非循环 sit 不再回卷到站立帧，而是停在最后一帧。请求新 clip 时 phase 会归零，并记录诸如 `idle->sit` 的 transition。

内置 office animation graph 位于 `stretch_mujoco/npc/animation/graph.py`，烘焙器和运行时共同读取这一份 clip/marker 定义，避免 manifest 生成规则与控制器完成条件各自硬编码后逐渐漂移。

clip 不存在时仍会选择 idle 保证画面可用，但只发出一次 `clip_fallback`，并通过 `NpcRuntimeState.animation_events` 暴露给观察者。这样 fallback 是可观测的降级，不会被上层误认为请求的 clip 已正常播放。

`PLAY_ANIMATION` 和 `INTERACTION_CUE` 可以通过 `completion_marker` 指定成功条件。SIT 使用 `seated` marker；只有动画 phase 真正越过该 marker，controller 才发出 `SUCCEEDED`。如果 production 配置错误、sit clip 缺失或 marker 永远未到达，命令最终走 deadline timeout，不会按固定逻辑时长提交落座。

### 9.3 NPC 物体附件与物理回执

新增文件：`stretch_mujoco/npc/attachment.py`

`AttachmentController` 为每个 NPC 独立拥有附件状态。ATTACH_OBJECT 的执行过程是：

1. 校验对象 body 存在且根 joint 为 free joint；
1. 校验 NPC 有 handover site；
1. 保存对象原 gravcomp 和 collision mask；
1. 附件期间关闭对象碰撞并令其跟随 NPC handover site；
1. 只有 `held_objects` 中确实出现对象后才返回成功。

DETACH_OBJECT 会把对象放到指定 MuJoCo site，恢复 gravcomp 和 collision mask，并确认附件记录消失后返回成功。`NpcSystem` 额外维护跨 NPC object claim，防止两个 NPC 同时附着同一个自由物体；失败或取消的 attach 会释放 claim。

这里采用的是确定性的运动学附件，而不是声称模拟了手指接触抓取。理由是当前 mesh-sequence NPC 没有可驱动手指关节；先建立可验证、可恢复的对象所有权和位姿链，比用视觉贴近冒充接触抓取更符合现有模型能力。

### 9.4 PICK_UP、PUT_DOWN 与语义提交

修改文件：

- `stretch_mujoco/agents/drivers.py`
- `stretch_mujoco/agents/action_recipes.py`
- `stretch_mujoco/agents/simulation_bridge.py`
- `stretch_mujoco/agents/runtime.py`

embodied driver 现在将：

- SIT 映射为等待 `seated` marker 的动画命令；
- PICK_UP 映射为 ATTACH_OBJECT；
- PUT_DOWN 映射为带目标 placement site 的 DETACH_OBJECT。

Runtime 会在 PUT_DOWN/HANDOVER 开始时把当前 held object 固化进不可变的 `ActionCommand.parameters`。这避免动作运行期间逻辑状态变化导致“放下了另一个对象”。和 MOVE_TO 一样，物理 receipt 成功前不会修改 `SemanticWorld.HOLDS`、对象 location 或 `EmployeeState.held_object`；失败、取消和超时也不会提交这些语义副作用。

轮询多阶段动作时，runtime 现在同步更新 driver handle。这个修复是 handover 必需的，否则第二阶段以后仍会反复查询第一条命令的回执。

### 9.5 通用 InteractionCoordinator

新增文件：`stretch_mujoco/agents/interactions.py`

`InteractionCoordinator` 不依赖具体机器人或 NPC，实现了一个小型参与者屏障：

- session ID 幂等；
- 显式 participants 和 object ID；
- 每个 phase 定义真正需要确认的参与者；
- 拒绝乱序确认和非参与者确认；
- 统一 succeeded、failed、cancelled、timed_out 终态。

handover 使用 `ready -> released -> received`，其中 ready 需要双方确认，released 只需要 giver，received 只需要 receiver。将要求按 phase 声明，而不是硬编码“所有阶段等所有人”，可以复用于 conversation、协作搬运或机器人交付，同时避免不相关参与者造成死锁。

### 9.6 robot→NPC 与 NPC→NPC handover

新增/修改文件：

- `stretch_mujoco/agents/robot_handover.py`
- `stretch_mujoco/agents/drivers.py`
- `stretch_mujoco/agents/runtime.py`

`RobotToNpcHandoverBridge` 定义了 robot→NPC 的接收边界。它要求机器人执行器先提供 `robot_release_confirmed=True`，然后才允许 NPC 提交 ATTACH_OBJECT；NPC 的成功回执越过 `received` 屏障后 session 才成功。这个布尔值是一个刻意显式的可信边界：当前项目的 `release_grasped_object()` 只是写 proxy，不构成物理释放回执，因此 bridge 不会自行把“发出释放请求”伪装成“释放已验证”。真实机器人或 MockRobot 的执行器需要在调用 bridge 前完成其释放验证。

NPC→NPC 复用相同 coordinator，driver 按以下顺序执行：

```text
giver ready cue receipt
    -> receiver ready cue receipt
    -> giver DETACH_OBJECT receipt
    -> receiver ATTACH_OBJECT receipt
    -> semantic HOLDS ownership transfer
```

任一阶段失败都会终止 session，且不会提前提交语义所有权。成功后 runtime 同时清空 giver 的 held object，并设置 receiver 的 held object，再验证目标 NPC 的状态与命令中固化的 object ID 一致。

### 9.7 删除旧全局状态并收窄兼容层

`MujocoServerProxies` 中的 `_humanoid_animation`、`_humanoid_sit_target`、`_humanoid_navigation_target` 和 `_humanoid_playback_speed` 已删除，服务进程不再维护第二套全局 humanoid 事实源。旧 public API 需要的临时 target 和 speed 只保存在客户端 adapter 内，并立即翻译成默认 `employee_01` 的 `NpcCommand`。

建议的最后一步同时带有“等所有调用方迁移后删除”的前置条件。检查发现 `examples/office_scene.py`、`examples/office_day_replay.py` 以及旧动画专项测试仍调用 `set_humanoid_*`，所以本轮保留这些 deprecated public adapter，避免破坏仍在仓库中的演示；它们已经不能绕开 `NpcSystem` 操纵服务器状态。

旧 `HumanoidMeshAnimator` 已不在在线服务器执行路径中，但源文件和专项兼容测试仍保留。下一轮应先把剩余示例改成显式 `npc_id + NpcCommand`，补等价场景回放测试，再删除 public adapter、旧 animator 和 legacy body-name adapter。换言之，全局运行路径已经删除，最后的源代码清理由调用方迁移前置条件保护。

### 9.8 第二轮验证结果

新增测试覆盖：

- sit 只在 `seated` marker 后完成；
- 缺失 clip 只发出一次 fallback event；
- attach 后对象 qpos 位于 NPC handover site；
- detach 后对象 qpos 位于 placement site；
- handover phase 必须按顺序确认；
- NPC→NPC handover 的四条物理命令顺序；
- robot→NPC 必须经过 release-confirmed 接口才提交接收附件。

实际执行结果：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests
217 passed in 37.63s

本轮修改文件 mypy --ignore-missing-imports
Success: no issues found in 10 source files

本轮修改文件 flake8
通过

production manifest 校验
Validated 2 NPC(s) and 1 bundle(s)

production population 生成并编译 MJCF
nmocap=2, ngeom=74
```

总体上，第一轮建立了“命令—控制器—回执—语义提交”的承重链；第二轮把落座、附件和交接真正接上这条链。仍需注意：mesh-sequence NPC 的附件是运动学约束，robot release confirmation 由机器人执行器提供，旧单 NPC API 还处在兼容迁移期。这些限制均被接口显式表达，没有以假成功隐藏。

## 10. 动作与动画优化：资产图与复合落座（第三轮，进行中）

本轮先实现后续动作优化共同依赖的最小边界，而未将尚无正式资产和完成条件的长时业务动作提前接入。

- `AnimationGraph` 现在由已通过 `NpcAssetManifest` 校验的 bundle clip 元数据构建：fps、loop、root motion 和 marker 由同一份资产事实源提供；`NpcSystem.from_population()` 可为每个 NPC 注入对应 graph。
- `AnimationEvent` 保留 marker 的 clip、phase、cycle 和 simulation time；`NpcRuntimeState` 继续提供简洁的事件名称投影，兼容既有 snapshot consumer。
- `SIT` 不再只发一条播放 sit 的命令。`MujocoNpcActionDriver` 现在将其下沉为 `MOVE_TO chair site -> ALIGN_TO seat yaw -> PLAY_ANIMATION(sit, seated marker)`；只有最后一条命令成功，runtime 才提交坐椅占用和逻辑 location。
- 座椅 yaw 归属于场景 action recipe 配置（当前 office scene 的两个座椅均为 pi），而不是 controller 内的默认值。缺少 site 或 yaw 会明确失败，避免面向错误方向仍报告落座成功。

新增测试验证 preview bundle 可派生 runtime graph，以及 SIT 的三个物理 receipt 全部成功前不会修改 semantic location。后续仍需为 stand up、work、eat、pick/place 手势提供正式资产、动作 recipe 和各自的持续/完成策略；当前没有把它们伪装成已具备物理完成保证。

### 10.1 提交前验证

本轮集成提交前重新执行了完整项目测试，结果为：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests
220 passed in 38.87s
```

对本次暂存文件执行 `uv run pre-commit run` 时，large-file、Black、isort、mdformat 和 codespell 均通过；flake8 仍被 `stretch_mujoco/mujoco_server.py` 与 `stretch_mujoco/stretch_mujoco_simulator.py` 中原项目已有的 unused import、bare except、无占位符 f-string、布尔比较和超长行阻断。新增文件中的 flake8 问题已修正，没有为了通过检查而修改这些与 NPC 改造无关的历史代码。

### 10.2 坐姿稳定态与起身过渡（当前状态）

`sit` 这一历史 clip 名称不再是 production animation contract 的一部分。正式
SMPL-X 烘焙器现在生成并登记以下三个明确角色不同的 clip：

- `sit_down`：非循环，由 `seated` marker 宣告已到达可提交座椅占用的姿态；
- `seated_idle`：循环坐姿保持；
- `stand_up`：非循环，由 `standing` marker 宣告可释放座椅占用。

`AnimationGraph` 是这两条完成后迁移的唯一 owner：`sit_down` 成功后请求
`seated_idle`，`stand_up` 成功后请求 `idle`。`NpcController` 只在带
`completion_marker` 的动画命令实际越过 marker 后调用该迁移；因此 mesh-sequence
后端没有使用 alpha crossfade 冒充 blend，也不会在动画刚发出时提前改变稳定态。

Agent 层新增 `ActionType.STAND_UP`。其 embodied driver 提交
`PLAY_ANIMATION(stand_up, standing)`；runtime 仅在成功 receipt 后删除
`OCCUPIED_BY`。如果 clip、marker 或 deadline 失败，座椅占用保持，供上层取消、重试或
恢复。新 production manifest 校验要求三个 clip 全部存在，旧的本地烘焙输出需要重新运行
baker 才能被当作 production bundle 使用；preview bundle 不会被替代或提升为 production。

本工作项的聚焦验证为：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_completion.py tests/test_action_driver.py tests/test_office_agents.py
22 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_assets.py::test_smplx_baker_writes_formal_production_clip_contract
1 passed

uv run mypy --ignore-missing-imports stretch_mujoco/npc \
  stretch_mujoco/agents/actions.py stretch_mujoco/agents/action_recipes.py \
  stretch_mujoco/agents/drivers.py stretch_mujoco/humanoid/smplx_animation_baker.py
Success: no issues found in 24 source files
```

在隔离 worktree 中运行 preview-manifest 测试仍会报告缺少已声明的
`assets/humanoid/cesium_man.png`。这说明预览资产未完整检出或未提供，不是校验应被放宽的
理由；本工作项没有修改 preview manifest、SHA 或 fallback 规则。

完整 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests` 未能进入测试执行：收集阶段
被当前开发环境缺少 `msgpack_numpy` 阻断，影响 grasp episode、GraspGen integration、OpenPI
与 ground-truth replay 的五个测试模块。这与本工作项无 import 交集；本次没有安装或修改
该可选抓取依赖。

### 10.3 动作 marker 与 production clip contract（当前状态）

production graph 现明确登记 `use_computer`、`pick_up`、`place`、`give`、`receive`、`talk`、
`gesture_wave` 与 `gesture_point`。这些 clip 由同一 `OFFICE_CLIPS` 供 baker、production
manifest 和 runtime graph 使用；runtime 从 bundle 建图时保留 graph policy 的 speed、safe
marker、interrupt 和预烘焙 overlay metadata。production bundle 缺少任一上述 clip 即校验失败。

`PICK_UP` 现在先播放 `pick_up` 并等待 `grasp` marker，之后才提交 `ATTACH_OBJECT`；
`PUT_DOWN` 对称地等待 `place/release` marker 后提交 `DETACH_OBJECT`。NPC—NPC handover 的
ready barrier 不再是 idle 的 duration cue：giver 先完成 `give/handover_ready`，receiver 再完成
`receive/handover_ready`，随后才沿既有 release/receive attachment receipt 链继续。故 marker
只是视觉—物理提交的前置条件，attachment receipt 仍是语义所有权提交的唯一依据。

`talk` 与 gesture 的 `upper_body_overlay` 字段仍只表示可选择预烘焙的组合 mesh sequence，
不表示 OBJ 后端支持实时骨骼层混合或 head look-at；这些能力仍在后续计划中。当前 focused
verification 为：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_assets.py::test_smplx_baker_writes_formal_production_clip_contract \
  tests/test_action_driver.py tests/test_animation_controller.py
8 passed in 0.42s
```

### 10.4 动画状态、速度和安全中断闭环（当前状态）

`AnimationState` 现在是 controller 所有的结构化观察值：它包含实际 resolved
clip、phase、有效 speed、loop、blend 和可选的预烘焙 overlay 标识；
`AnimationLifecycle` 则由同一 controller 沿 `requested -> navigating -> aligning -> playing -> completed/failed` 更新。命令的 accepted receipt 不再被视为播放或完成：MOVE_TO
与 ALIGN_TO 分别公开 navigating/aligning，PLAY_ANIMATION 在 backend 真正采样后才公开
playing，marker/attachment 失败、取消或 deadline 会公开 failed。终态会保留到下一次 intent，
不会被后台 idle frame 的采样重写为 playing。

loop clip 的初相可由注入 seed 决定；相同 seed 得到相同 phase。phase 推进现在实际乘以
`ClipDefinition.speed`，因此 graph policy 的速度元数据不再只是声明。SAFE_MARKER 请求会
记录 `interrupt_deferred` 并只在声明 marker 后切换；UNINTERRUPTIBLE 非循环 clip 会运行至
终点，cancel/deadline 则使用受控的 force-idle recovery。找不到请求 clip 时仍会产生
`clip_fallback` 以保持观察性，但该 embodied command 立即返回
`FAILED(reason=clip_unavailable:<clip>)`，不会以 fallback idle 和 duration 伪造成功。

OBJ mesh-sequence backend 当前只声明 `mesh_sequence`/`phase_switch`，没有已验证的透明
crossfade 或骨骼 layer-blend capability。因此 `blend` 保持 0，transition 通过 marker-safe
hard cut 完成；`upper_body_overlay` 仅表示选择已烘焙的全身组合 clip，并不宣称运行时上半身
叠加或 head look-at。真正 opacity crossfade 必须先有同 topology、可共存的 frame mesh，且
在资产/材质契约中显式声明 alpha blend 后再实现。

本轮聚焦验证：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_animation_controller.py tests/test_npc_completion.py \
  tests/test_action_driver.py \
  tests/test_npc_assets.py::test_smplx_baker_writes_formal_production_clip_contract
16 passed in 0.46s

uv run mypy --ignore-missing-imports stretch_mujoco/npc/animation \
  stretch_mujoco/npc/controller.py stretch_mujoco/agents/action_recipes.py \
  stretch_mujoco/agents/actions.py stretch_mujoco/agents/drivers.py \
  stretch_mujoco/humanoid/smplx_animation_baker.py
Success: no issues found in 11 source files
```

同次运行完整的 animation/action/assets focused set 时，17 个相关测试通过；两个
`tests/test_npc_assets.py` preview-manifest 测试在收集资产时失败，因为隔离 worktree 缺少
已由 manifest 声明的 `assets/humanoid/cesium_man.png`。没有为通过测试放宽 SHA/存在性校验。

### 10.5 导航与脚步 marker 同步（当前状态）

`LocomotionController` 仍是唯一写入 NPC mocap root pose/yaw 的组件，但现在公开直线路由的
`route_revision`、`replan_attempt`、`route_tangent`、progress timeout 和失败原因。MOVE 命令先
采样 walk，只有跨过 `left_foot` 或 `right_foot` marker 后才开始 root motion；到达位置和目标 yaw
后同样等待下一个 foot marker 才返回成功。运行中的 MOVE 取消会返回 running，直到 foot marker
安全停止并发出原 command 的 cancelled receipt。

无进展会在 recipe/command 指定的 `progress_timeout` 后增加 route revision；达到
`max_replans` 后以 `FAILED(reason=route_blocked)` 结束并 idle recovery。site 在运行中失效则以
`route_invalid` 失败。两种原因以及 route/replan/stop-marker 投影到 `NpcRuntimeState`，不提交
逻辑位置或其他语义副作用。

本轮公开 headless 覆盖 MOVE 的脚步 marker 起停和一次有限重规划后失败：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_completion.py tests/test_animation_controller.py tests/test_action_driver.py
19 passed in 0.44s
```

当前 route abstraction 仍是直线路径；它不包含导航网格避障或动态障碍物绕行。下一轮可在不改变
controller/root-pose ownership 的前提下替换 waypoint provider，并复用 revision/progress/failure
contract。

### 10.5 Replay phase offsets and validated embodied recipes (current state)

`NpcSystem` now accepts a stable `simulation_seed` (including the explicit, stable
default used by `from_model`). Each controller derives loop offsets with SHA-256 over
the simulation seed, NPC ID, command ID, resolved clip, and cycle; it does not use
Python's randomized hash or wall-clock entropy. Runtime state projects `phase_seed`,
`phase_offset`, `last_marker`, `pending_clip`, and `deferred_interrupt` while retaining
the existing snapshot fields, so older snapshots remain readable.

`ActionRecipe` is now an immutable contract with target/site, yaw, marker, timeout,
recovery, and observation fields. The first complete recipe path is `USE_COMPUTER`:
the driver refuses incomplete configuration before command submission, then executes
`MOVE_TO -> ALIGN_TO -> PLAY_ANIMATION(use_computer, computer_cycle)`. No talk/session
state, crossfade, or production visual asset was added; actions without a complete
recipe must not be treated as physically supported.

Focused verification after this change:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_animation_controller.py tests/test_action_driver.py tests/test_npc_completion.py
16 passed in 0.44s

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_npc_assets.py::test_smplx_baker_writes_formal_production_clip_contract
1 passed in 0.39s
```

The two preview-manifest tests remain blocked by the absent declared preview asset
`assets/humanoid/cesium_man.png`; this work does not alter asset validation.

### 10.6 Canonical seat transitions reconciled (current state)

The action recipe projection now names `sit_down` rather than the retired `sit`
clip, matching the driver and graph. A successful `seated` marker requests
`seated_idle`; a successful `standing` marker requests `idle`. The runtime accepts
and verifies `STAND_UP` only for the agent occupying the target chair, and releases
`OCCUPIED_BY` only after the terminal physical receipt. The safe-marker interrupt
contract remains owned by `AnimationController`; no crossfade capability was added.

Focused reconciliation verification:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_animation_controller.py tests/test_action_driver.py \
  tests/test_npc_completion.py tests/test_office_agents.py
29 passed in 0.47s
```

The wider runtime mypy invocation still reports pre-existing Optional-target and
untyped-driver errors in `agents/runtime.py`; its relevant behavior is covered by the
focused tests above, and this reconciliation does not suppress or alter those checks.
