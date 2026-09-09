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

对于当前由 `_write_material_texture()` 生成的 SMPL-X 平色 atlas，另有 `flat_layers.py`。它只识别生成器定义的 skin/shirt/pants/shoes RGB 区域，并生成透明、UV 对齐的编辑层；它不是通用的人体语义分割器。示例 `office_warm_base_v1` 的 layer spec 和 recipe 位于 `assets/humanoid/generated/animations/appearance_sources/office_warm_base_v1/`，可依序运行：

```bash
uv run make_npc_flat_layers --spec .../office_warm_base_v1/layers.json --output-dir .../office_warm_base_v1/layers
uv run bake_npc_appearance --recipe .../office_warm_base_v1/recipe.json \
  --output-dir .../appearances/office_warm_base_v1 \
  --asset-manifest .../animations/manifest.json --bundle smplx_office_neutral_v1
```

`hair_layers.py` 复用已烘焙 idle OBJ 的顶点与 UV，只栅格化高于设定 head height 的头部面，生成短发发帽层。当前共享来源为 `appearance_sources/short_hair_v3/hair.json`，并登记可复用的 `short_brown_v3` layer；它只改变同一短发覆盖区域的颜色，不能增加发丝、长发轮廓或物理摆动。生成与登记命令为：

```bash
uv run make_npc_short_hair_layers --spec .../short_hair_v3/hair.json --output-dir .../short_hair_v3/layers
uv run bake_npc_appearance --recipe .../short_hair_v3/recipe.json \
  --output-dir .../appearances/office_short_brown_freckles_v1 \
  --asset-manifest .../animations/manifest.json --bundle smplx_office_neutral_v1
```

### 3.0.1 可组合身份 catalog（当前事实源）

文件：`stretch_mujoco/npc/appearance_pipeline/catalog.py`

外观的权威输入不再是人口配置中散落的最终 PNG 路径，而是 versioned appearance catalog。catalog 只拥有四类事实：base atlas 及其 SHA-256、可重用 layer 的 category/path/SHA-256、稳定 visual identity，以及 identity 对应的最终 `appearance_id`。layer 可以被多个 identity 共用，identity 则固定列出 layer 顺序和可审计 traits；生成的 `body.png`、sidecar 和 MJCF material 都是可删除投影。

```text
catalog identity + layer hashes
    -> bake_npc_identity
    -> content-addressed recipe receipt / final body.png
    -> NPC asset manifest appearance
    -> MuJoCo scene material
```

`NpcEmbodiment` 仍保留 `appearance` 以兼容现有 scene builder 和 runtime，但在 population 设置 `appearance_catalog` 后必须额外设置 `visual_identity`。解析会加载 catalog，并拒绝 identity 不存在、layer/base 被篡改，或 identity 产生的 `appearance_id` 与 embodiment 不一致的配置。这样 agent profile 的姓名/岗位与可渲染身份解耦，而 visual identity 在重建时仍是确定的。

本地生产示例 catalog 位于 `assets/humanoid/generated/animations/appearance_catalog.json`，当前包含 Alex 的暖肤色办公装、Morgan 的棕色短发、黑色/深金/赤褐色短发，以及雀斑、眉毛/胡须、2D 圆框眼镜 identity。该 catalog 与派生 PNG 一样位于被忽略的本地 SMPL-X 输出目录，不能作为可再分发资产提交；应从受许可的输入和 recipe 重新生成。

```bash
uv run bake_npc_identity \
  --catalog stretch_mujoco/models/assets/humanoid/generated/animations/appearance_catalog.json \
  --identity office_warm_base_v1 \
  --output-dir stretch_mujoco/models/assets/humanoid/generated/animations/appearances/office_warm_base_v1 \
  --asset-manifest stretch_mujoco/models/assets/humanoid/generated/animations/manifest.json \
  --bundle smplx_office_neutral_v1
```

当前 catalog 支持任意顺序的同 UV 2D layers（例如 skin、freckles、face_detail、hair、top、bottom、shoes）；它不承诺防止两个绘制者选择语义冲突的 layer。该类组合策略由内容制作约束管理。2D 眼镜只是脸部贴花，长发、真实镜框、帽子和背包仍属于未来 `slots.py` 的独立 mesh 工作。

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

包含 `office_overview` camera 的原生办公室录制场景使用近距离、陡俯视的 free camera（lookat `(0, 0.15, 0.75)`、distance `6.4`、azimuth `90`、elevation `-63`），而不使用场景中位于室外的低角度 `office_overview` fixed camera。场景开口在南侧；在入口正视的 `azimuth=180` 基础上，相机围绕场景中心逆时针旋转 90° 至 `azimuth=90`。该调整只改变水平朝向，保持距离和高度不变，并避免墙面遮挡；不改变 MuJoCo 世界状态或 snapshot 位姿。

3-D renderer 的录制信息从整列左侧状态栏改为顶部半透明窄条：它显示时间、每个 NPC 的 ID/动作和最新事件。这样状态信息仍可检查，但不再遮挡画面左侧或在底部叠加多行字幕。

2-D `OfficeMp4Recorder` 现在也先写 OpenCV `mp4v` 临时文件，再以与 3-D renderer 相同的 FFmpeg 参数转码：`libx264`、`yuv420p`、`+faststart` 和无音轨。无论通过 `render_topdown_video()`、`tools/render_office_video.py`，还是示例直接使用 recorder，最终 MP4 都应可由 VS Code 预览；缺少 `ffmpeg` 会明确失败，不会留下伪装为可预览的 `mp4v` 结果。

NPC 外观验收统一由 `tools/render_npc_acceptance_video.py` 执行：在未改动的办公室 MJCF 原生光照与家具环境中，先从前、后、左、右、上五个近距离视角播放 `walk` 动画帧，再用同一五个视角播放 `sit` 动画帧。为了在不把相机移出办公室的前提下保持全身可见，验收渲染仅在内存中将 review camera 的 FOV 扩至至少 65°；不添加灯光、不改变办公室家具、环境、材质或 MJCF 文件。JSON 报告以 `acceptance_standard: npc_appearance` 标记用途，并逐视角记录相机参数、覆盖到的动画帧、动画是否确实跨帧变化和遮挡结果。除颜色/材质稳定外，预检还要求每帧拥有与首帧一致的 body/OBJ 配饰槽位，且原始 alpha 只能为稳定的 0 或 1；每次切换后当前帧都必须完全不透明。任一缺失槽位、分数 alpha、颜色变化、单帧动画或遮挡都会使 `passed` 为 false。`sit` 不依赖 review-only 椅子 site；它仍从同一未改动办公室 MJCF 的 NPC frame sequence 动态播放，用于满足外观任务的坐姿检查。

验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../stretch_mujoco/.venv/bin/python -m pytest -q tests/test_npc_acceptance_video.py` 通过（7 passed）；以 `MUJOCO_GL=egl` 对组合办公室和 `npc_alex_chen` 的五视角短渲染返回 `passed: true`，每个视角都覆盖 walk 帧 0 与 1 且无可见性失败。常规 pytest 自动加载 ROS `launch_testing` 时仍会因环境缺少 `lark` 在测试收集前失败。

验证：`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../stretch_mujoco/.venv/bin/python -m pytest -q tests/test_mp4_recorder.py tests/test_recording.py` 通过（6 passed in 3.21s），其中 recorder 测试用 `ffprobe` 断言最终流为 `h264`/`yuv420p`。直接运行 pytest 会被环境自动发现的 ROS `launch_testing` 插件阻塞，原因是该环境缺少 `lark`，并非此渲染器测试失败。

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

`LocomotionController` 现在在包含 `office_floor` 的办公室 MJCF 中从当前 collision geometry 建立 `OfficeNavigationMesh` 路径，并逐 waypoint 使 mocap root 朝 route tangent 转向。进度超时触发的 `max_replans` 不再只是增加计数：每次 retry 都重新读取当前 MuJoCo 几何并重建 route；无法生成路径则以 `route_unavailable` 失败并恢复 idle。没有 `office_floor` 的最小协议 fixture 明确保留直线路径，用于不加载办公室几何的单元测试，不能被解释为生产避障。

步行动画的相位速率现在直接使用 `LocomotionController.speed`：导航变快或变慢时，walk clip 的播放速率同向缩放，保持 root 位移、朝向和脚步 marker 属于同一个速度契约；非行走 clip 仍只使用各自 recipe 定义的播放速度。

两条 OBJ frame-sequence 播放路径（兼容 `HumanoidMeshAnimator` 和多 NPC 的 `MeshSequenceBackend`）采用确定性的逐帧硬切换：切 clip 时旧 frame 立即隐藏、新 frame 立即显示。`AnimationState.blend` 保留为 skinned backend 的未来接口，OBJ backend 始终报告默认值 `0.0`；不再以 alpha 交叉淡化模拟骨骼混合或上半身 overlay。

`ActionType` 已声明 `talk`、`gesture_point` 与 `gesture_wave`。三者均由 `MujocoNpcActionDriver` 下沉为 `MOVE_TO participant approach site -> ALIGN_TO configured yaw -> PLAY_ANIMATION clip/marker`，并要求参与者 target、site、yaw、clip、marker 和 recipe timeout；play payload 还携带 `gaze_target` 与 `upper_body_overlay` 意图。当前逐帧 OBJ backend 不声明 `upper_body_overlay` capability，因此 controller 会以 `overlay_backend_unavailable` 拒绝该 play stage 并恢复 idle，不能将它当成已实现的注视/局部手势。要启用该路径，必须为 scene 提供 participant approach sites，并接入有该 capability 的 skinned backend。

`MujocoNpcActionDriver` 现在把每个 embodied stage 的 recipe timeout 写入 `NpcCommand.deadline`。PICK_UP 必须先从 object→human approach-site 配置发出 MOVE_TO，再播放 `pick_up` 并越过 `grasp` marker 后 ATTACH；PUT_DOWN 同样先到 location 的 human approach site，再播放 `place`、越过 `release` marker 并 DETACH 到 placement site。HANDOVER 必须配置 receiver 的 handover approach site，先由 giver 到达该 site，再执行 `give/receive` marker 和 attachment barrier；缺少任一必需 site 会在提交前以稳定的 prepare error 拒绝，而不会在当前位置伪造交互成功。

workflow 的取消和超时现在也会回收其内部 stage 状态：HANDOVER 的 interaction deadline 触发时会取消当前 MuJoCo command，并以 `TIMED_OUT/deadline_exceeded` 返回；PICK_UP、PUT_DOWN、USE_COMPUTER 和 HANDOVER 传播底层 command 的 `TIMED_OUT`、`CANCELLED` 或 `FAILED`，而非把三者混为普通失败。取消进行中的 workflow 会取消其当前 stage 并移除 workflow，后续同一 execution ID 不会读到陈旧回执。

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

## 11. 外观组合与身份 catalog（第四轮，进行中）

本轮在 `appearance_pipeline/catalog.py` 建立了“一个事实源、多个投影”的外观边界。`AppearanceCatalog` 对 base atlas、每个可组合 layer 和 visual identity 都记录并验证 SHA-256；identity 以稳定 ID、ordered layer IDs、traits 和最终 `appearance_id` 表示。`bake_npc_identity` 依据 catalog 生成最终 `body.png`，再复用既有资产 manifest 登记流程；它不会修改 MuJoCo runtime 的材质数据，也不会允许未校验的磁盘文件在运行时热加载。

population schema v2 兼容保留 `embodiment.appearance`，新增可选 `visual_identity` 与顶层 `appearance_catalog`。一旦声明 catalog，每个 NPC 必须声明 identity，且解析会确认该 identity 正好生成该 NPC 的 `appearance`；这使旧 population 无需迁移即可继续运行，而生产配置可以把人名/岗位和可审计的 visual identity 关联起来。当前 catalog 只保留 Alex 的暖色办公装基础、细框眼镜身份和 Morgan 的短发雀斑身份；先前方向错误的短发和面部贴图 identity 已清除，不能再被选用。

当前可组合的 category 是内容元数据而不是渲染 slot：`skin`、`top`、`bottom`、`shoes`、`hair` 均仍合成为单一 `body` 纹理。catalog 允许将 future `freckles`、`face_detail` 等同 UV layer 加入 identity；它不会把长发、眼镜或帽子错误宣称为可实现的轮廓变化，那些仍要求未来的 mesh `slots.py`。

本轮实际验证使用当前本地 `.venv`，因为 `uv` 重建项目包时需要从 PyPI 下载 `uv-build`，但当前网络连接被重置：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q \
  tests/test_npc_appearance_catalog.py tests/test_npc_schema.py \
  tests/test_npc_appearance_bake.py tests/test_npc_hair_layers.py \
  tests/test_npc_flat_layers.py tests/test_npc_assets.py tests/test_npc_scene_builder.py
18 passed in 1.68s

.venv/bin/mypy --ignore-missing-imports stretch_mujoco/npc/appearance_pipeline stretch_mujoco/npc/schema.py
Success: no issues found in 7 source files

catalog bake + production population/asset validation + generated MJCF compilation
baked=alex_warm_v1 identities=5 mocap_bodies=2
```

`mdformat`、`codespell` 和 `git diff --check` 也通过。本轮没有将本地受限 SMPL-X 派生 atlas、catalog 或 PNG 提交；它们仍在 ignored `generated/` 下，必须从已获许可的源资产生成。完整 `tests/` suite 在本轮未作为完成证据，因为工具窗口没有返回该长运行进程的终态；上述 focused tests 覆盖新增边界。

## 12. 正式语义 mask 与面部身份细节（第五轮，进行中）

`appearance_pipeline/semantic_masks.py` 将生成器的平色 atlas 和已烘焙 idle OBJ 转换为正式、独立的 `skin.png`、`top.png`、`pants.png`、`shoes.png`、`hair_cap.png` 和 `face.png`。它同时写入 `semantic_masks.json`，记录 topology、base/mesh 来源与每个 mask 的 SHA-256。catalog 现在可声明该 manifest 及其 SHA-256，并逐项复验 mask 文件；因此被篡改的 mask、与 catalog 不同 topology 的 mask 或缺失 mask 都会在烘焙前失败。

`face.png` 仅代表由 head height、半径、前向位置和 outward normal 推导的 face surface，并非人工美术标注的通用人脸分割。`face_details.py` 所有输出都被这个 mask 裁剪；当前保留可重复生成的 `freckles_light_v4`、`brows_soft_brown_v4`、`beard_chin_brown_v4` 和 `glasses_thin_round_v4` layer。最后一项被明确标为 `accessory_2d`：它是无轮廓的贴图视觉效果，不参与碰撞，也不替代未来的独立眼镜 mesh。

```bash
uv run make_npc_semantic_masks --spec .../semantic_masks_v2/spec.json \
  --output-dir .../semantic_masks_v2/masks
uv run make_npc_face_details --spec .../face_details_v4/spec.json \
  --output-dir .../face_details_v4/layers
uv run bake_npc_identity --catalog .../appearance_catalog.json \
  --identity office_warm_glasses_chin_beard_v1 --output-dir .../appearances/office_warm_glasses_chin_beard_v1 \
  --asset-manifest .../manifest.json --bundle smplx_office_neutral_v1
```

当前已烘焙的细节组合为 `office_warm_glasses_chin_beard_v1` 和 `office_short_brown_freckles_v1`。当前仍未实现 facial-photo projection、真人身份复刻或 3D 配饰；内容作者需要在统一 UV 上制作受许可 layer，并为需要轮廓的配饰等待 geometry slot。

本轮验证使用本地 `.venv`（`uv` 的网络重建限制仍见第 11 节）：20 个外观、catalog、schema、asset 和 scene-builder 相关测试通过；`mypy --ignore-missing-imports stretch_mujoco/npc/appearance_pipeline stretch_mujoco/npc/schema.py` 通过；production population 和 manifest 校验通过，生成场景可被 MuJoCo 编译，结果为 `identities=8 masks=6 detail_identities=3 mocap_bodies=2`。`mdformat`、`codespell` 和 `git diff --check` 通过。

## 13. 现有人设的配饰组合与离屏复核（第六轮，进行中）

production population 现在选择两个可追溯的组合：Alex Chen 使用 `office_warm_glasses_chin_beard_v1`，Morgan Lee 使用 `office_short_brown_freckles_v1`。人物名称只归属 profile；identity、appearance 和 layer 名称只描述可复用的外观组合，因此任一 NPC 都可选择任一组合。没有在运行时修改材质，也没有覆盖旧输出。

`face_details.py` 现在按 face mask 的每个实质性、不连通 UV 岛分别绘制细节。先前将多个岛合并为一个边界框，会把眼镜等图案画进透明 UV 接缝；新测试确保双岛 mask 的两侧都获得贴图内容。历史上用于局部复核的 `tools/render_npc_personas.py` 会创建 standalone MJCF 并添加审查灯光和地面，因此已在统一真实场景验收后删除；当前唯一受支持的外观验收入口是 `tools/render_npc_acceptance_video.py`，必须加载组合办公室场景并保留原生环境。

首次审查发现旧 face mask 把局部正脸方向反选为 `Y >= 0`，配饰落在头后；旧短发只按高度选择，覆盖了鼻子和耳朵。该输出不再作为推荐身份：新增 face mask v2 使用 `Y <= -0.02` 且 outward normal 的 Y 分量不大于 `-0.35` 选择正脸；短发 v3 额外要求 upward normal，且将最低高度提高到 `1.67m`。重新生成、烘焙并离屏复核后，Alex 的眼镜、眉毛和胡须在正脸，Morgan 的发帽止于额头上方。雀斑是低不透明度贴图，当前审查灯光下较淡。

第二次审查又收窄了 facial decal：仅向不小于主 face island 一半面积的岛绘制，避免把胡须和眼镜重复到 neck/seam island；胡须从口部下移为下巴短胡须，镜圈缩小为细框，并加入贴图式鼻梁和镜腿线。正脸复核中嘴唇不再被胡须覆盖。镜腿在正视时天然不明显，且仍仅为 UV 像素，不会形成侧面可见的实体镜架或碰撞形状。需要可见轮廓、物理碰撞或更精细发型时，仍应引入独立 geometry slot，而不是把 UV decal 宣称为 3D 配饰。

在用户明确要求清理后，所有经渲染证明无效或只绑定旧人物名称的 appearance、face detail、short hair 和 semantic mask v1–v3 资源已从本机的 ignored `generated/` 目录移入系统回收站。catalog 与 manifest 同步收缩：当前 production appearance 是 `office_warm_base_v1`、`office_warm_glasses_chin_beard_v1`、`office_short_brown_freckles_v1`；来源保留 `semantic_masks_v2`、`short_hair_v3`、`face_details_v4`。删除的是可重新生成的本地派生文件，不影响版本库中受跟踪的代码或基础资产。

## 14. 独立 OBJ 配饰（第七轮，进行中）

`NpcEmbodiment.accessories` 现在接受通用 accessory ID。asset manifest 的 bundle 可声明每个 accessory 的 OBJ 与逐 clip/逐 frame anchor JSON；二者都由 SHA-256 校验。scene builder 为人体每个 clip frame 同时生成同名帧的 accessory geom，`MeshSequenceBackend` 因而会在切换 idle、walk、sit 帧时同步切换人体和配饰，不会把眼镜固定在 mocap 根节点。

原 `cap_simple_v1` 是一次性低多边形 demo，已删除其生成器、基线 manifest 注册与派生 OBJ。它不能作为运行时配饰或后续资产命名模板；正式帽子统一通过下面的 recipe-bound `baseball_cap_v1` 路径构建。

## 15. 外部服装素材预处理（本地 preview）

`appearance_sources/jogging_melange_v1/` 现包含从用户提供的 jogging-melange PBR 包提取的 diffuse map 生成的 `top_jogging_melange_v1.png`。预处理将纹理在 UV canvas 平铺，并以正式 `semantic_masks_v2/masks/top.png` 写入 alpha，得到 `1024×1024` 的 top-only RGBA layer；`source_receipt.json` 记录输入 archive/member、UV topology、处理方式、layer 和 thumbnail 的 SHA-256。检查确认 layer alpha 与 top mask 完全一致（598571 个非零 alpha 像素）。

该输入未随包提供可审计许可证或来源 URL，receipt 因而标记 `asset_quality: preview` 与 `license_status: user_supplied_unverified`。它尚未写入 appearance catalog、asset manifest 或 production population，运行时不会加载；获得许可证信息、完成图案比例和 UV seam 的人工审查后，才可将其登记并烘焙为 production appearance。

曾用于局部试验的 `office_jogging_cap_preview_v1`、`cap_simple_v1` 及相应 preview manifest 已清除；它们不是 production 外观或运行时配置。离屏 MP4 仍只能作为本地验收证据，不构成 production 资源登记。

## 16. 主工作区共享资源库与分支投影（当前事实源）

本机 NPC 资源只有主工作区一个物理所有者。`aaa_workspace/raw_resources/` 是未审核输入的暂存区，运行配置不得引用；审核后仍需本地保存的来源归档进入 `stretch_mujoco/models/assets/humanoid/sources/`，受许可证约束的模型与动作输入进入 `private/`，经过校验的 manifest、纹理、OBJ、anchor 和 receipt 投影进入 `generated/`。版本控制中的 recipe、runtime config、selection example 和 README 只描述重建契约，不复制受限资源。

其他 worktree 使用以下命令建立投影：

```bash
.venv/bin/python tools/link_npc_shared_assets.py \
  --shared-project-root <main-project-root> \
  --worktree-root <other-worktree-root>
```

该命令链接主工作区的 `sources/npc`、`generated/animations`、`private/{amass_intake,smplx,uv}`，并补齐主工作区存在而分支缺少的模型 payload；README、XML、Python 和已由 Git 跟踪的公共模型保持分支本地。因为非主 worktree 的根 `aaa_workspace` 已是指向主工作树的符号链接，raw staging 天然共享：链接器在两端 `raw_resources` 解析为同一路径时直接跳过逐项投影。只有根 workspace 尚未共享时才逐项创建 raw-resource link。命令仍拒绝覆盖真实目录、内容不同的真实文件或指向其他位置的链接，因此分支若已有新资源，必须先做来源、许可证、内容哈希和运行验收，再明确放入主工作区的相应所有者目录，然后删除分支副本并重新建立链接。不得把分支私有副本当作新的事实源。

AMASS 的唯一可复现入口是 `prepare_amass_npc_motion`。`list` 以流式方式检查本地 NPZ、目录或 `.tar.bz2`；`prepare` 只接受 schema-v1 selection，校验 SMPL-X `pose_body`、帧区间、FPS 和有限数值，输出选定的 `body_pose` 及 canonical `transl`，并记录许可证、源成员 SHA-256、裁剪/重采样参数和输出 SHA-256 的 receipt。canonical translation 将首帧钉定为 NPC runtime anchor，丢弃水平 source displacement，仅保留相对 SMPL-Y 高度，以避免 walk 移动 NPC，同时允许坐/起类动作保留必要的垂直高度变化；缺少 `trans` 的旧 source 写入零 translation，因而保持兼容。restricted baker 与 review renderer 使用相同的 identity root orientation、canonical translation 和一次 reference-ground offset，且仍必须显式提供 `--motion-root`、缺少完整 `OFFICE_CLIPS` 时失败，不再生成伪造的示例动作。

OBJ/GLB 配饰以 `models/accessories/<accessory_id>.recipe.json` 与 `<accessory_id>.runtime.json` 为受版本控制事实源；资源 ID 使用“用途 + 版本”，而导入来源、NPC 名称和 preview 状态不得进入 ID。来源 archive 固定从共享 `sources/npc` 读取，派生输出固定写回共享 `generated/animations`。`prepare_obj_accessory.py`、`fuse_obj_accessory.py` 和 `build_npc_fused_accessory.py` 共同执行源格式规范化、逐帧 head-follow 融合、receipt 与严格 manifest/population preflight；receipt 是审计投影，不能替代 recipe。

本轮验证结果：三套开发 worktree 重复运行链接命令均无新增输出，每套有 `486` 个资源链接、`0` 个断链，且全部解析到主工作区，三套 `git status --short` 均为空；cap source archive 和 jogging texture archive 的 SHA-256 均与现有 receipt 一致。`validate_npc_assets.py --population stretch_mujoco/models/office_population.production.example.json` 通过（2 NPC、1 bundle）；资源处理与 schema 的 40 个聚焦测试通过；`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests` 为 `263 passed`。新增 Python 文件的 Black、flake8 与 mypy 检查通过，`git diff --check` 通过。未设置 EGL 的同一 tests 命令仅在 renderer 测试因无 `DISPLAY`/`gladLoadGL` 失败；设置 `MUJOCO_GL=egl` 后该测试及全套均通过。当前环境未安装 mdformat/codespell，未将其宣称为已运行通过。

资源命名规范化后，正式配饰 ID 采用“用途 + 版本”：`baseball_cap_v1` 取代导入过程名 `cap_source_v1`；其受版本控制的 `recipe`/`runtime`、共享 source archive、`runtime/`、runtime population 和 runtime manifest 使用同一 ID，正式 manifest 不再带 `preview` 后缀。`cap_simple_v1`、`receipt_*_runtime`、`recipe_runtime*`、旧 `cap_source_v1` runtime projection 和旧 preview manifest 均为无引用实验投影，已移入系统回收站；基线 manifest 已移除 `cap_simple_v1` 注册。当前 runtime 的目标为 `npc_alex_chen`；使用 `.venv/bin/python tools/build_npc_scene.py --accessory-runtime-config stretch_mujoco/models/accessories/baseball_cap_v1.runtime.json --output stretch_mujoco/models/.npc_alex_chen_baseball_cap_office.xml --include-base-scene` 会先从最新 recipe 重建帽子、anchor、融合帧、runtime manifest 和 runtime population，再构建办公室场景。随后对 `runtime_populations/baseball_cap_v1/npc_alex_chen.json` 运行 `validate_npc_assets.py` 的当前结果是 `Validated 10 NPC(s) and 1 bundle(s)`。对涉及该历史文件的 Black 检查仍报告 `tools/prepare_obj_accessory.py`、`tests/test_prepare_obj_accessory.py` 的基线格式化差异；本次未做无关整文件格式化。

生产外观 roster 现由受版本控制的 `models/appearance_recipes/office_personas_v1.roster.json` 与 `tools/build_npc_persona_roster.py` 重建。它生成 4 种 skin、4 种 short-hair colour、4 套 top、3 套 bottom、3 双 shoes，并结合现有 freckles、brows、chin beard 与 2D glasses layer；每个 identity 固定 seed、生成 body atlas、thumbnail 和 metadata receipt。`office_population.production.example.json` 现在是 10 人配置：原 `employee_01`/`employee_02` 迁移为 `npc_alex_chen`/`npc_morgan_lee`，其余八位有独立 profile、agent_id、visual identity 和完整的 `skin`/`hair`/`top`/`bottom`/`shoes`/`accessories`/`scale` 配置。Alex 的 `office_sage_beard_glasses_v1` identity 已使用其 `hair_brown_v1` 短发 layer；当前 `baseball_cap_v1` 的融合 runtime 则将帽子以逐帧 body OBJ 的方式加到 Alex，不能被每帧 mesh 的局部重心变化甩离头部。验证为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_npc_appearance_bake.py tests/test_npc_appearance_catalog.py tests/test_npc_schema.py tests/test_npc_production_appearance_roster.py tests/test_accessory_recipe.py tests/test_npc_acceptance_video.py`（此前为 `25 passed`）和 `tools/validate_npc_assets.py --population stretch_mujoco/models/office_population.production.example.json`（`Validated 10 NPC(s) and 1 bundle(s)`）。每人接受 office-native front/walk、side/walk、seated/work 的 120-frame acceptance render；报告要求无 material/alpha failures 且每个镜头 `occlusion_passed: true`。验收 renderer 现在会只将目标 NPC 的 sequence-frame geom 设为可见，同时隐藏同一已加载 office MJCF 中其他 roster NPC 和历史 `humanoid_preview` 的 frame geom；它不更改 office XML、家具、灯光、环境或目标 NPC。这样共享 desk spawn 的重叠人体不会遮挡检查对象；report 的 `review_visibility.policy=target_npc_frames_only` 与 `hidden_npc_ids` 记录这一 review-only 可见性策略。当前 10 个 120-frame report 均 `passed=true`、无 material/alpha failures，并记录了该策略。

`appearance_pipeline/textile_layers.py` 建立织物 intake 到 UV layer 的受控边界。versioned textile spec 为每个 source ZIP 固定 SHA-256、来源 URL、license、ZIP 内 diffuse member、semantic mask、tile 宽度和 seed；生成器只读取已审核并移入 `assets/humanoid/sources/npc/textures/` 的 archive，拒绝 hash 不符或越出 generated asset root 的输出，并在 `appearance_sources/.../textiles/textile_layers.receipt.json` 记录 source/mask/output hashes。`office_personas_v1` 当前将 jersey melange 与浅灰 jogging melange 分别应用到 Olivia 的 `top_jersey_melange_v1`、Wei 的 `top_jogging_melange_v1`；gingham 资源保留为可审计输入，但不再由 production Jordan 使用，因为它在当前 atlas 上会产生明显形变。jogging layer 使用 `top.png` 语义 mask，Wei 的 bottom 保持 `bottom_charcoal_v2`。`build_npc_persona_roster.py` 会先重建这些层再生成 catalog、body atlas、thumbnail 与 manifest。

面部细节的全络腮胡现在不再从二维 face atlas 猜测鬓角位置：每个 `beard_full` style 都必须声明由 `semantic_masks_v2` 的已烘焙 OBJ 生成的 `sideburn_mask`，缺失该字段会明确失败。该 mask 仅选择头部耳前的侧向表面（高度、横向距离和前向 `Y` 均受限）；`beard_full_black_v1` 与保留兼容名称的 `beard_full_black_sideburns_v2` 都引用同一 mask，独立可复用资源为 `sideburns_black_v1`，不再有 Marco 专属配置。这样任一 persona 应用该全胡或鬓角资源都会使用相同的 UV/mesh 语义定位。重新生成语义 mask、face-detail layers 和全员 roster 后，Marco Silva 与 Wei Zhang（当前两个 full-beard identity）各自以组合办公室的原生光照完成 200-frame、前/后/左/右/顶部单人验收；两份 report 均为 `passed=true`、`lighting=scene_native_only`、每镜头 `animation_passed=true` / `occlusion_passed=true`，且 `color_failures`、`transparency_failures` 为空。该轮 focused 检查为 `PYTHONPATH=. ../stretch_mujoco/.venv/bin/python -m pytest -q tests/test_npc_face_details.py tests/test_npc_semantic_masks.py tests/test_npc_flat_layers.py tests/test_npc_textile_layers.py`，结果 `17 passed in 0.83s`。

Lena Fischer 的 `office_olive_goatee_glasses_v1` 使用独立的 `goatee_dark_brown_lena_v2` layer；其 `vertical_offset=0.070` 相比默认 `0.055` 仅向下移动胡型，未修改其他 goatee identity。`face_details.py` 对 goatee 的该可选值强制要求数值且限制在 `[0.0, 0.12]`，同时移动其下巴胡和连带小胡子，避免两者分离。重新烘焙 roster、组合办公室并对 Lena 生成 200-frame 单人验收后，report 为 `passed=true`、`lighting=scene_native_only`，五个动态镜头均无颜色、透明度或遮挡失败；同一 focused pytest 命令复跑结果为 `17 passed in 0.72s`。

Jordan Patell 不再使用会在当前 atlas 上产生明显形变的 `top_gingham_check_v1`。其 production identity 改为 `office_burgundy_boxed_beard_analytics_v2`，保留橄榄肤色、赤褐短发、海军蓝下装、棕色鞋和方形胡须，仅将上装换成已验证的纯色 `top_burgundy_v1`；population 的 `appearance`、`visual_identity` 与显式 `appearance_config.top` 同步为该新 identity，避免 name/config/catalog 漂移。重新烘焙 identity、编译组合办公室后，Jordan 的 200-frame 单人办公室验收 report 为 `passed=true`、`lighting=scene_native_only`，五个动态镜头无颜色、透明度或遮挡失败。`tests/test_npc_flat_layers.py tests/test_npc_textile_layers.py` 为 `6 passed in 0.42s`，`tests/test_npc_production_appearance_roster.py` 为 `2 passed in 0.43s`，`tools/validate_npc_assets.py --population stretch_mujoco/models/office_population.production.example.json` 验证 `10 NPC(s) and 1 bundle(s)`。

当前 10 人 no-hair review 已由受版本控制的 `models/appearance_recipes/office_personas_no_hair_v1.lock.json` 固定。lock 记录 roster、production population、appearance catalog、asset manifest、office scene 和 baseball-cap runtime config 的 SHA-256；为每位 NPC 固定 appearance/visual identity/显式 slots、已烘焙 `body.png` SHA-256，以及当前 no-hair MP4 与 acceptance report SHA-256。其 build contract 明确只以 `baseball_cap_v1.runtime.json` 加入场景、包含原始办公室，并排除 `beautiful_hair_v1` 与 `ponytail_hair_v1`。当前每份 no-hair MP4 均为 400 帧：五个动态 `walk` 方位镜头加五个动态 `sit` 方位镜头。`tools/validate_no_hair_npc_lock.py --lock stretch_mujoco/models/appearance_recipes/office_personas_no_hair_v1.lock.json --workspace-root <workspace>` 会校验全部 source/body/video/report/scene 哈希、10 人 population 对应关系、每份 report 的 NPC ID、`passed=true` 及精确的五个 walk 加五个 seated/sit 方位镜头，并校验场景中不包含被排除的发型 ID。任一外观、烘焙贴图、no-hair 场景或当前验收 artifact 的漂移都会失败而非静默替换。

用户已明确批准的 `baseball_cap.glb`、`beautiful-hair-1.zip` 与 `ponytail-hair-for-character.zip` 已从 intake 提升到共享 `sources/npc`，并由分别面向 Alex、Jordan、Priya 的 `baseball_cap_v1`、`beautiful_hair_v1`、`ponytail_hair_v1` recipe/runtime 配置接入 production。runtime schema v2 声明 source format（`glb` 或 nested ZIP OBJ）、Y-up 单位换算、嵌套成员和每个发型的 source vertical anchor；recipe schema v4 以 `mesh_scale`、`lateral_offset_m`、`back_offset_m`、`head_clearance_m`、`back_tilt_degrees`、`roll_degrees`、`yaw_degrees` 持久化统一缩放和完整六自由度位姿。v1–v3 缺少的新字段在读取时补零，保持历史配置兼容。这三项第三方源的前方约定与 NPC 相反，均固定 `yaw_degrees=180.0`，在 Y-up→Z-up 规范化之后绕 MuJoCo Z 轴旋转。GLB 在内存中解码为 world-space OBJ，原始 source hash、规范化 mesh hash、anchor hash 和 recipe hash 都写入 receipt/manifest。全部三者仍使用同一、无缓存的 build path：每次 `build_npc_scene.py --accessory-runtime-config ...` 都从当前 recipe 重新生成 OBJ、anchors、fused frames、runtime manifest 和 population。

`tools/tune_npc_obj_accessory.py` 是 recipe 的交互编辑投影。它从一份 production runtime config 解析 source accessory、base manifest 和指定 clip/frame，调用与 `prepare_obj_accessory.py` 正式构建完全相同的 source-to-head transform 和 head anchor 函数，在 Matplotlib 3-D 窗口中以蓝色配饰/棕色人体实时更新。界面提供 scale、X/Y/Z、pitch/roll/yaw 七个滑块、front/left/back/right/top 视角、鼠标 orbit/zoom、Reset 与原子 `Save recipe`；关闭而不保存不会修改配置。为保证拖动延迟可控，显示面按确定性索引降采样，但位姿计算使用全部源顶点。预览不会在每次拖动时重建整套动画；保存后仍必须走正式 `build_npc_scene.py --accessory-runtime-config ...`，由严格 manifest/population preflight 完成所有 clip 的融合。`--snapshot` 提供无显示环境的可重复 smoke render。

本轮验证命令为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg .venv/bin/pytest -q tests/test_accessory_recipe.py tests/test_prepare_obj_accessory.py tests/test_fuse_obj_accessory.py tests/test_tune_npc_obj_accessory.py tests/test_npc_production_appearance_roster.py tests/test_npc_textile_layers.py tests/test_npc_appearance_catalog.py tests/test_npc_appearance_bake.py tests/test_npc_schema.py tests/test_npc_acceptance_video.py`，结果 `40 passed`。以 `baseball_cap_v1.runtime.json --snapshot /tmp/baseball_cap_accessory_tuner_v4.png --maximum-faces 3000` 成功生成 `211 KiB` 的真实 cap/head 预览。三份 v4 production recipe 重新组合场景后，runtime 校验为 `Validated 10 NPC(s) and 4 bundle(s)`；三份生成 receipt 均包含并等于 recipe 的七个姿态字段。

旧实现改写共享 base bundle，会使后续合成的配饰可被全员引用。现在每个 recipe 都复制 source bundle 为 `smplx_office_neutral_v1__fused__<accessory_id>`，只把目标 NPC 的 embodiment bundle 改到该副本；base bundle 和其他 NPC 的 clip frame 路径完全不变。多个 `--accessory-runtime-config` 按顺序把上一阶段的 manifest/population 传给下一阶段，因此同一 office scene 可同时保留这三位 NPC 的独立逐帧头随动配饰。配饰面使用 identity-selected body-atlas UV，避免引入未经声明的透明材质；这不是单独的 accessory PBR material 管线。

本次校验：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_accessory_recipe.py tests/test_prepare_obj_accessory.py tests/test_fuse_obj_accessory.py tests/test_npc_production_appearance_roster.py tests/test_npc_textile_layers.py tests/test_npc_appearance_catalog.py tests/test_npc_appearance_bake.py tests/test_npc_schema.py tests/test_npc_acceptance_video.py` 得到 `38 passed`；production population 校验为 `Validated 10 NPC(s) and 1 bundle(s)`，三套融合 bundle 的 runtime population 校验为 `Validated 10 NPC(s) and 4 bundle(s)`。以组合 office MJCF 对 Alex、Jordan、Priya 各运行一次 120-frame front/side/seated acceptance renderer，三份 report 均为 `passed=true`、`lighting=scene_native_only`、每个 shot `occlusion_passed=true`，且 `color_failures`、`transparency_failures` 均为空。生成 MP4 和 report 均为本地 review artifact，不提交版本库。

## 17. NPC 对话会话与 mock 机器人（第八轮，当前事实）

`stretch_mujoco/agents/conversation.py` 的 runtime-owned `ConversationSession` 记录 session ID、两个参与者、topic、turn、started_at、timeout、status、transcript 与 interrupt policy。P0/P1 将协议固定为 schema v1：受限的 session/terminal/interrupt/participant-kind/turn-policy/event/error-code 类型都在 conversation module 中定义；session 与 turn ID 使用协调器内的确定性递增格式，重复占用同一参与者的非终态 session 会被拒绝。逻辑/mock 路径直接进入 `ACTIVE`；`REQUESTED/APPROACHING/ALIGNING` 为 action workstream 的实体接入预留，尚不宣称实际靠近或 talk cue。`EmployeeState.availability` 收敛为 `available`、`executing`、`in_conversation`、`blocked`，并增加有界的 `conversation_id`、`social_energy` 与 `stress`；schema-v1 的 `busy` 输入只在状态构造边界映射为 `executing`，不扩展旧 API。

启动会话不再直接信任松散的 `objects[*].position/yaw`。只读 `ConversationPerception` adapter 统一 live `SemanticWorld.pose_snapshot()`（`time`、`objects`、quaternion，adapter 推导 yaw）、offline recording（`sim_time`、`agents`、yaw）及无时间戳的 legacy objects/yaw 输入。它拒绝超过 runtime 同一时钟 0.5 的陈旧/未来 observation、缺参与者、缺 yaw/quaternion、NaN/无穷位姿和显式不可见 participant；legacy 无时间戳 payload 只在边界按“现在观测”处理，保证旧调用可迁移但不能得到陈旧检测。P0 未改变 MuJoCo/action driver；`ALLOW`、`SAFE_MARKER`、`IMMEDIATE` 仅是合同值，runtime 仍拒绝打断任何 busy embodied execution，真实 pause/cancel receipt 留给 action workstream 合同。

会话只允许 NPC↔NPC 或 NPC↔Stretch robot；开始后将 NPC 标为 `in_conversation`、锁定 attention，状态机和 utility planner 不会推进其原计划。默认 round-robin 的 `expected_speaker` 防止连续抢话；turn/session deadline 都会终止会话。complete/cancel/fail/timeout 共用幂等 terminal cleanup：只首次写入 RuntimeEvent/AgentMemory，恢复保存的 attention、availability 和 goal，保留原 action queue、reservation 与 execution ID；busy embodied action 对 `ALLOW` 也仍拒绝，直到 action workstream 有安全取消 receipt。robot 会话的 `request` 仅经 `validate_action(REQUEST_ROBOT)` 后创建关联 `RobotTask`，而且该逻辑 RPC 不会覆盖会话锁；`clarify`/`acknowledge` 只可引用已接受 task，`handover_confirm` 需要 task 成功以及独立的 robot release、NPC attachment、interaction barrier receipt。重复 task/receipt 不会二次 semantic commit 或 event；mock robot 可确定性模拟延迟、失败、丢失和重复回执。NPC↔NPC 会话允许 `greeting`、`progress_inquiry`、`meeting_invitation` 和 `conflict_resolution`；`NpcConversationScheduler` 以 creation time、双方 ID 稳定排序，同时争用时只准一个 session，记录拒绝/cooldown，meeting invitation 仅保存候选/接受/拒绝/过期状态，绝不直接改日程或 table reservation。LLM dialogue prompt 仅能返回候选 intent/text，`OfficeAgentRuntime` 才会记录候选；运行时执行敏感字检测、160 字符上限、speaker cooldown 及 intent 对话类型校验，并在文本无效、过长或含敏感词时使用 intent 对应的固定模板。

测试使用不加载 MuJoCo visual assets 的 deterministic fixture（3 NPC、`stretch_3`、稳定 observation clock/pose/yaw 与可控制的 `MockRobotExecutor`），覆盖协议 ID、状态兼容、距离/朝向边界、陈旧/缺 yaw/NaN observation、幂等终态、轮流发言/turn timeout、busy `ALLOW` 拒绝、task/receipt handover、mock 失败/丢失/重复回执、3 NPC 稳定排序/无死锁/cooldown 和 meeting invitation 响应。P4 仍未驱动真实接近、转身或 talk marker。

本轮验证（conversation worktree）为：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/sxu/xs/stretch_mujoco/.venv/bin/pytest -q tests/test_conversation.py tests/test_office_agents.py tests/test_interactions.py tests/test_action_driver.py tests/test_npc_completion.py` 得到 `46 passed in 0.49s`；`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/sxu/xs/stretch_mujoco/.venv/bin/pytest -q tests` 得到 `243 passed in 38.23s`；conversation/models/actions/mock_robot 的 Black 与 mypy（`--ignore-missing-imports`）以及 `git diff --check` 通过。最初 `tests/test_recording.py` 与完整 suite 因新 worktree 缺少 ignored MuJoCo texture/preview asset 投影失败；根因是链接器在已共享的 root `aaa_workspace` 内再次逐项链接 raw resources，安全保护提前终止，导致后续 model payload 阶段未执行。链接器现以 resolved-path identity 识别该情形并跳过多余 raw projection；重新投影创建 `470` 个模型链接、`0` 个断链，第二次运行创建 `0` 个链接，随后 P0 完成时 `MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/sxu/xs/stretch_mujoco/.venv/bin/pytest -q tests` 为 `235 passed in 38.51s`。对整个 `runtime.py` 的 mypy 仍报告既有 ActionCommand/driver/robot-task optional 类型错误（不在 P0 修改的行）；未用忽略项掩盖。

`examples/npc_conversation_acceptance.py` 现在提供无 LLM、确定性的 P0--P3 验收回放。它从同一组 runtime-generated JSONL snapshots 与 RuntimeEvent 生成 `conversation_2d.mp4` 和 `conversation_3d.mp4`，同时写入 manifest 与 `conversation_acceptance.json`；后者明确标记为 logical conversation replay，不能被解释为实体接近、转身或 talk marker receipt。storyboard 覆盖 3 个 NPC 的 greeting、progress inquiry、meeting invitation（接受）与 conflict resolution，以及 NPC--Stretch 的 request、clarify、acknowledge、handover confirm、RobotTask 成功和三项 handover receipt。2-D 回放将 active `conversation_id` 投影为紫色状态、attention target 连线和意图化事件文字；3-D 回放用 `build_native_multi_npc_scene(..., employee_numbers=(1, 2, 3))` 从原有 `office_scene.xml` 复制已有 animated humanoid，保留该办公室原有灯光、家具和 Stretch，再由 MuJoCo 离屏渲染并转码 H.264。录制生成物均为 ignored local artifact，不提交；第三个 NPC 是用于回放的同一预览人体克隆，不是新增 production appearance asset。

复现：`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python examples/npc_conversation_acceptance.py --output-dir /tmp/npc-conversation --render both --fps 5 --width 640 --height 360`。本轮 focused `tests/test_conversation_acceptance.py tests/test_recording.py tests/test_mp4_recorder.py tests/test_conversation.py` 为 `30 passed in 2.45s`；完整 `MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests` 为 `245 passed in 40.44s`。本轮 `mypy --ignore-missing-imports` 覆盖演示和录制源时，仅报告已有 OpenCV type stub 未声明 `cv2.VideoWriter_fourcc` 的两处错误；运行时渲染已实际成功，未用 ignore/suppression 掩盖该类型桩问题。

发布范围现冻结为“单 NPC 向 Agent/mock robot 发任务接口”。`OfficeAgentRuntime.submit_action(ActionCommand(..., ActionType.REQUEST_ROBOT, ...))` 是不依赖多人对话的直接调用入口；对话场景中的 `request_robot_task` 仍复用同一验证、reservation 和 `RobotTask` 提交路径。`RobotTask.robot_id` 现明确记录归一化后的目标 robot ID，snapshot-v2 的 `robot_tasks[*]` 同时输出 task ID、robot ID、conversation ID、请求者、物品、目标位置和状态，使外部 demo/bridge 可稳定关联任务与回执。`examples/npc_conversation_acceptance.py --scenario robot-task` 是发布验收用的窄范围演示：仅创建 `employee_01` 与 `stretch_3` 的一次 deliver 请求，顺序展示 request、clarify、acknowledge、mock robot 成功、handover receipt 和 confirm；它写入 `npc_robot_task_acceptance.json`、同一 JSONL 驱动的 2-D/3-D MP4 和 native-office scene。该录制仍是逻辑 task replay，不宣称 NPC 已实体靠近、转身或说话。多人社交调度与完整 `--scenario full` 回放保留为已验证的后续能力，均不再是当前发布阻塞项。新增 focused `tests/test_conversation_acceptance.py tests/test_office_agents.py tests/test_recording.py tests/test_conversation.py` 为 `42 passed in 2.60s`；在 `MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 下完整 `tests/` 受执行时限分成无重叠两组，分别 `143 passed in 21.75s` 与 `103 passed in 21.37s`，合计 `246 passed`。以同一 EGL 环境实际生成 640×360、7 帧的 H.264 `npc_robot_task_2d.mp4` 和 `npc_robot_task_3d.mp4`，并人工检查首帧的单 NPC/Stretch 状态与事件叠层。

## 18. BABEL 标注驱动的 AMASS 动作候选（当前状态）

`stretch_mujoco/humanoid/babel_intake.py` 新增 `prepare_babel_npc_candidates`，它只读取本地 BABEL v1.0 release ZIP 与既有 AMASS candidate index，不解压或复制原始动作。它只支持 BABEL 中与本地 AMASS stage-II member 一一对应的 BMLmovi 与 CMU 路径，校验该 member 已在 index 中以 SMPL-X 技术可用状态登记，并把 BABEL 逐帧秒级边界转换为 source FPS 帧区间。候选仅接受精确的、空白规范化后的标签；宽泛的 `gesture`、`transition` 和 `interact with/use object` 不会被静默映射为 office clip。

本地 BABEL 处理生成 ignored、`pending_visual_review` 的 provenance candidate record，而不是 intake selection 或 baker input。当前 BMLmovi 可由精确 BABEL frame label 覆盖 `idle`、`walk`、`sit_down`、`stand_up`、`gesture_wave`、`gesture_point`、`pick_up`、`place`、`talk` 与 `receive`；仍缺 `seated_idle`、`work`、`use_computer`、`eat` 和 `give`，因此 restricted production bundle 仍必须保持不完整失败，不能宣称视觉动作完成。BABEL 全集元数据已定位到单一 CMU AMASS source archive 中的 typing、eat、give 和 seated 候选；在该 archive 按其许可取得并完成同样的 index/visual review 前，不将其加入 selection。

本地 CMU archive 已完成 bzip2 全流校验（解压 9,210,675,200 bytes；压缩 archive SHA-256 `5a5207d1968740164b2487bdbeaf644046a63b09984ac2baaa72916a0b333df3`；解压字节流 SHA-256 `b566adec09d9fc3b440079248b50de00ce76b668ee089dd133281579f28d7225`），并以 ignored `raw_resources/CMU.candidates.jsonl` 登记 2,079 个 NPZ。与 BABEL 交叉后，精确标签候选覆盖 14/15 个 `OFFICE_CLIPS`；CMU 自身只缺 `receive`，但已有 BMLmovi 精确候选覆盖该 clip。`type motion` 只被并列列为 `use_computer` 和 `work` 的候选，不是批准的语义重用。BMLmovi 的 10 个和 CMU 的 13 个去重 source-frame range 均已使用与 baker 相同的 canonical root policy 重渲染，receipt SHA 已全部复核。坐标/大位移显示问题已经消除，但候选质量审查拒绝当前 BMLmovi `sit_down`（深蹲后回站）及 CMU `sit_down`/`stand_up`（没有可提交的椅子过渡）。额外登记的 Transitions archive（111 NPZ，110 个技术可用）有 BABEL 精确标注的同一 `sit_stand` 序列，但真实帧是站立到地面深蹲、再从地面起身，也被拒绝为 chair transition。当前三套本地授权来源均没有可自动批准的 `sit_down`/`stand_up`；`pick_up`、`place`、`receive`、`give` 及 typing/work 仍因物体/桌椅语义不可从无环境人体图中自动批准。所有候选继续为 `pending_visual_review`，没有 selection、baker input、manifest 或 production 完成声明；完整 production bundle 必须保持失败。

Focused verification:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_babel_intake.py tests/test_amass_intake.py tests/test_amass_motion_review.py
14 passed

uv run black --check stretch_mujoco/humanoid/babel_intake.py tests/test_babel_intake.py
passed

uv run flake8 stretch_mujoco/humanoid/babel_intake.py tests/test_babel_intake.py
passed

uv run mypy stretch_mujoco/humanoid/babel_intake.py
Success: no issues found in 1 source file
```

## 19. NPC 场景轨迹 profile（当前事实）

`stretch_mujoco/npc/trajectory_profiles/office_v1.json` 是办公室 NPC 行动轨迹的受版本控制规范，schema v1 将稳定的场景语义与易变的几何投影分开：每个 anchor 固定 `site` 与 `role`，每条有向 route 固定 `from`、`to` 和允许衔接的 action 类型；profile 同时钉定目标 MJCF 的 SHA-256。profile 不保存世界坐标 waypoint；家具移动、尺寸变化或导航参数变化后，旧坐标会失效，必须由当前 MuJoCo collision geometry 重新求解。

当前 `office_v1` 已预先确认四条可行路线：左右工作位到会议区、会议区到零食台、零食台到储物柜。运行以下命令会加载 MJCF、确认 profile 的 scene 名称、确认全部 site/`office_floor` 存在，并用 `OfficeNavigationMesh` 对每条 route 做碰撞预检：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python tools/validate_npc_trajectories.py \
  --scene stretch_mujoco/models/office_scene.xml \
  --profile stretch_mujoco/npc/trajectory_profiles/office_v1.json \
  --receipt /tmp/office_v1_trajectory_receipt.json
```

receipt 中的 waypoint 是此次构建的审计投影，不能作为后续场景或运行时的事实源；运行时 `LocomotionController` 仍会从实时 geometry 重规划。新场景应先提供明确的、面向人的 interaction/navigation site，再复制并按新场景创建独立的版本化 profile，更新该场景的 SHA-256，最后把上述 preflight 加入该场景的 CI。不存在 anchor、floor 不符合导航约定、profile 与 scene 名或 SHA-256 不匹配，或任一 route 不可达时，命令均会以稳定错误码失败，禁止把未确认路线发布给 NPC 行为层。

preflight 还会用 `LocomotionController` 逐 10 ms 推进每条 route，包含每个渐进转向姿态；MuJoCo 必须报告 NPC collision proxy 与所有非 `office_floor` 场景 geom 零接触，否则以 `trajectory_route_collision` 拒绝。`office_v1` 当前分别审计了 336、315、554、674 个 move/turn pose，均为零接触；JSON receipt 对每条 route 写入采样数与 `collision_free: true`。

可用以下命令生成实际 controller 驱动的路线巡回视频。四条 route 依次由真实 `MOVE_TO` 命令执行；路线之间的 source 切换明确标为静态 placement，不会伪造穿模移动。renderer 在开始写 MP4 前强制运行上述路径与转向碰撞审计，并把每条 route 的审计采样数写入画面 HUD 和 sidecar report：

```bash
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python \
  tools/render_npc_trajectory_tour.py \
  --scene stretch_mujoco/models/office_scene.xml \
  --profile stretch_mujoco/npc/trajectory_profiles/office_v1.json \
  --output /tmp/npc_all_routes.mp4
```
