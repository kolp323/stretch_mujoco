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

### 14.1 不透明 hair-card 与 SMPL-X UV 兜底（当前事实）

`models/accessories/*.recipe.json` 的 schema v5 将 OBJ 的渲染处理纳入同一 recipe
事实源：`render_policy.double_sided` 要求融合器为每个 accessory 三角形额外写出反向
绕序面；可选 `render_policy.surface_fallback`（当前支持 `smplx_head_uv_v1`）以 recipe
声明的发色与头部阈值，从 reference body OBJ 的真实 UV 三角形生成目标 NPC 专属的不透明
atlas 和 mask。它解决开放、单面 hair cards 在运行时背面剔除或缝隙中显出内部头皮的问题，
但不改变 OBJ 轮廓，也不能替代近距离发型建模。

该能力只沿既有 `recipe -> build_npc_fused_accessory -> runtime manifest/population ->
build_npc_scene` 路径工作。构建器复制原 visual identity 的 catalog entry，唯一改变其
`appearance_id` 为 fallback atlas 的派生 appearance，并将 runtime population 的目标 NPC
指向该派生 identity；因此不会为绕过 fallback 而移除 `appearance_catalog` 或降低其 hash/
layer 校验。atlas、mask、逐帧融合 OBJ、runtime manifest/population 和 catalog projection
均为可删除的 recipe 投影；来源 OBJ、基础 atlas、源 manifest/population/catalog 均只读。

2026-09-11 已以 `ponytail_hair_v1` 在 `npc_priya_narayanan` 上完成本路径验证：严格
资产校验覆盖 10 个 NPC/2 个 bundle；`render_npc_acceptance_video.py` 对生成的 runtime scene
输出 40 帧十视角验收视频且 `passed=True`。验收产物仅保留在本地
`aaa_workspace/demo/hair_acceptance/`，不作为受版本控制的资源输入。

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

`appearance_pipeline/textile_layers.py` 建立织物 intake 到 UV layer 的受控边界。versioned textile spec 为每个 source ZIP 固定 SHA-256、来源 URL、license、ZIP 内 diffuse member、semantic mask、tile 宽度和 seed；生成器只读取已审核并移入 `assets/humanoid/sources/npc/textures/` 的 archive，拒绝 hash 不符或越出 generated asset root 的输出，并在 `appearance_sources/.../textiles/textile_layers.receipt.json` 记录 source/mask/output hashes。`office_personas_v1` 当前将 jersey melange 与浅灰 jogging melange 分别应用到 Olivia 的 `top_jersey_melange_v1`、Wei 的 `top_jogging_melange_v1`；gingham 资源保留为可审计输入，但不再由 production Jordan 使用，因为它在当前 atlas 上会产生明显形变。新增的 bi stretch、caban、cotton jersey、crepe georgette、两种 denim、fabric leather、poly wool herringbone、rough linen、stretch poplin 与 velour velvet 已各自注册为可选 `top` layer；它们不改动当前十名 production NPC 的 identity。jogging layer 使用 `top.png` 语义 mask，Wei 的 bottom 保持 `bottom_charcoal_v2`。`build_npc_persona_roster.py` 会先重建这些层再生成 catalog、body atlas、thumbnail 与 manifest。

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

本轮验证（conversation worktree）为：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_conversation.py tests/test_office_agents.py tests/test_interactions.py tests/test_action_driver.py tests/test_npc_completion.py` 得到 `46 passed in 0.49s`；`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests` 得到 `243 passed in 38.23s`；conversation/models/actions/mock_robot 的 Black 与 mypy（`--ignore-missing-imports`）以及 `git diff --check` 通过。最初 `tests/test_recording.py` 与完整 suite 因新 worktree 缺少 ignored MuJoCo texture/preview asset 投影失败；根因是链接器在已共享的 root `aaa_workspace` 内再次逐项链接 raw resources，安全保护提前终止，导致后续 model payload 阶段未执行。链接器现以 resolved-path identity 识别该情形并跳过多余 raw projection；重新投影创建 `470` 个模型链接、`0` 个断链，第二次运行创建 `0` 个链接，随后 P0 完成时 `MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests` 为 `235 passed in 38.51s`。对整个 `runtime.py` 的 mypy 仍报告既有 ActionCommand/driver/robot-task optional 类型错误（不在 P0 修改的行）；未用忽略项掩盖。

`examples/npc_conversation_acceptance.py` 现在提供无 LLM、确定性的 P0--P3 验收回放。它从同一组 runtime-generated JSONL snapshots 与 RuntimeEvent 生成 `conversation_2d.mp4` 和 `conversation_3d.mp4`，同时写入 manifest 与 `conversation_acceptance.json`；后者明确标记为 logical conversation replay，不能被解释为实体接近、转身或 talk marker receipt。storyboard 覆盖 3 个 NPC 的 greeting、progress inquiry、meeting invitation（接受）与 conflict resolution，以及 NPC--Stretch 的 request、clarify、acknowledge、handover confirm、RobotTask 成功和三项 handover receipt。2-D 回放将 active `conversation_id` 投影为紫色状态、attention target 连线和意图化事件文字；3-D 回放用 `build_native_multi_npc_scene(..., employee_numbers=(1, 2, 3))` 从原有 `office_scene.xml` 复制已有 animated humanoid，保留该办公室原有灯光、家具和 Stretch，再由 MuJoCo 离屏渲染并转码 H.264。录制生成物均为 ignored local artifact，不提交；第三个 NPC 是用于回放的同一预览人体克隆，不是新增 production appearance asset。

复现：`MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python examples/npc_conversation_acceptance.py --output-dir /tmp/npc-conversation --render both --fps 5 --width 640 --height 360`。本轮 focused `tests/test_conversation_acceptance.py tests/test_recording.py tests/test_mp4_recorder.py tests/test_conversation.py` 为 `30 passed in 2.45s`；完整 `MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests` 为 `245 passed in 40.44s`。本轮 `mypy --ignore-missing-imports` 覆盖演示和录制源时，仅报告已有 OpenCV type stub 未声明 `cv2.VideoWriter_fourcc` 的两处错误；运行时渲染已实际成功，未用 ignore/suppression 掩盖该类型桩问题。

发布范围现冻结为“单 NPC 向 Agent/mock robot 发任务接口”。`OfficeAgentRuntime.submit_action(ActionCommand(..., ActionType.REQUEST_ROBOT, ...))` 是不依赖多人对话的直接调用入口；对话场景中的 `request_robot_task` 仍复用同一验证、reservation 和 `RobotTask` 提交路径。`RobotTask.robot_id` 现明确记录归一化后的目标 robot ID，snapshot-v2 的 `robot_tasks[*]` 同时输出 task ID、robot ID、conversation ID、请求者、物品、目标位置和状态，使外部 demo/bridge 可稳定关联任务与回执。`examples/npc_conversation_acceptance.py --scenario robot-task` 是发布验收用的窄范围演示：仅创建 `employee_01` 与 `stretch_3` 的一次 deliver 请求，顺序展示 request、clarify、acknowledge、mock robot 成功、handover receipt 和 confirm；它写入 `npc_robot_task_acceptance.json`、同一 JSONL 驱动的 2-D/3-D MP4 和 native-office scene。该录制仍是逻辑 task replay，不宣称 NPC 已实体靠近、转身或说话。多人社交调度与完整 `--scenario full` 回放保留为已验证的后续能力，均不再是当前发布阻塞项。新增 focused `tests/test_conversation_acceptance.py tests/test_office_agents.py tests/test_recording.py tests/test_conversation.py` 为 `42 passed in 2.60s`；在 `MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 下完整 `tests/` 受执行时限分成无重叠两组，分别 `143 passed in 21.75s` 与 `103 passed in 21.37s`，合计 `246 passed`。以同一 EGL 环境实际生成 640×360、7 帧的 H.264 `npc_robot_task_2d.mp4` 和 `npc_robot_task_3d.mp4`，并人工检查首帧的单 NPC/Stretch 状态与事件叠层。

conversation 工作流随后以开发主分支 `feat/npc-system@4541df0`（`merge(npc): integrate appearance optimization`）为新基线完成 rebase，重写后的 5 个对话提交为 `f8c8480`、`df53bd5`、`f650ccc`、`487d801`、`786294b`。外观验收 HUD 与对话 HUD 的唯一代码冲突按组合语义解决：保留开发主分支的紧凑顶部状态栏及 close/unoccluded 相机测试，同时保留 `attention_target` 显示与第三 NPC 可编译测试。conversation worktree 的 production population 和 `office_personas_v1.roster.json` 与主工作区逐字节一致，`validate_npc_assets.py` 结果为 `Validated 10 NPC(s) and 1 bundle(s)`。联合聚焦测试为 `59 passed in 4.19s`；完整 `tests/` 分成无重叠两组运行，分别为 `178 passed in 22.57s` 与 `135 passed in 25.43s`，合计 `313 passed`。rebase 后再次实际生成单 NPC task demo，2-D/3-D 各 7 帧，3-D 为 H.264 640×360。

## 18. BABEL 标注驱动的 AMASS 动作候选（当前状态）

`stretch_mujoco/humanoid/babel_intake.py` 新增 `prepare_babel_npc_candidates`，它只读取本地 BABEL v1.0 release ZIP 与既有 AMASS candidate index，不解压或复制原始动作。它只支持 BABEL 中与本地 AMASS stage-II member 一一对应的 BMLmovi 与 CMU 路径，校验该 member 已在 index 中以 SMPL-X 技术可用状态登记，并把 BABEL 逐帧秒级边界转换为 source FPS 帧区间。候选仅接受精确的、空白规范化后的标签；宽泛的 `gesture`、`transition` 和 `interact with/use object` 不会被静默映射为 office clip。

本地 BABEL 处理生成 ignored、`pending_visual_review` 的 provenance candidate record，而不是 intake selection 或 baker input。当前 BMLmovi 可由精确 BABEL frame label 覆盖 `idle`、`walk`、`sit_down`、`stand_up`、`gesture_wave`、`gesture_point`、`pick_up`、`place`、`talk` 与 `receive`；其中 `gesture_point` 已按 2026-09-14 用户授权完成 CMU/BABEL sequence `3182` 的 production projection。除该已登记动作外，仍缺 `seated_idle`、`work`、`use_computer`、`eat` 和 `give`，因此这些动作仍不能宣称视觉动作完成。BABEL 全集元数据已定位到单一 CMU AMASS source archive 中的 typing、eat、give 和 seated 候选；在该 archive 按其许可取得并完成同样的 index/visual review 前，不将其加入 selection。

本地 CMU archive 已完成 bzip2 全流校验（解压 9,210,675,200 bytes；压缩 archive SHA-256 `5a5207d1968740164b2487bdbeaf644046a63b09984ac2baaa72916a0b333df3`；解压字节流 SHA-256 `b566adec09d9fc3b440079248b50de00ce76b668ee089dd133281579f28d7225`），并以 ignored `raw_resources/CMU.candidates.jsonl` 登记 2,079 个 NPZ。与 BABEL 交叉后，精确标签候选覆盖 14/15 个 `OFFICE_CLIPS`；CMU 自身只缺 `receive`，但已有 BMLmovi 精确候选覆盖该 clip。`type motion` 只被并列列为 `use_computer` 和 `work` 的候选，不是批准的语义重用。BMLmovi 的 10 个和 CMU 的 13 个去重 source-frame range 均已使用与 baker 相同的 canonical root policy 重渲染，receipt SHA 已全部复核。坐标/大位移显示问题已经消除，但候选质量审查拒绝当前 BMLmovi `sit_down`（深蹲后回站）及 CMU `sit_down`/`stand_up`（没有可提交的椅子过渡）。额外登记的 Transitions archive（111 NPZ，110 个技术可用）有 BABEL 精确标注的同一 `sit_stand` 序列，但真实帧是站立到地面深蹲、再从地面起身，也被拒绝为 chair transition。当前三套本地授权来源均没有可自动批准的 `sit_down`/`stand_up`；`pick_up`、`place`、`receive`、`give` 及 typing/work 仍因物体/桌椅语义不可从无环境人体图中自动批准。`gesture_point` 已有 experiment selection、baker input、production manifest 和派生 bundle projection；其余候选继续为 `pending_visual_review`，没有 production 完成声明；完整 production bundle 仍需其他缺失动作闭包后才能作为整体宣称完成。

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

### 18.1 已验收动作的 production 登记（当前事实）

2026-09-10 用户确认此前已审核通过 `stand_up`、`put_down` 和 `gesture_wave`，并明确授权 production 注册。这是对第 18 节自动候选审查结果的人工验收决定，而不是把尚未审查的 queue 条目直接标为完成。受限本地 intake approval record 固定三项来源和裁剪：BMLmovi Subject 11 `1005:1536` → `stand_up`，Subject 50 `573:1085` → `place`（`PUT_DOWN` 的 runtime clip），以及 Subject 14 `37:287` → `gesture_wave`；全部按 8 FPS canonical root policy 生成可复现 baker input。

`bake_and_register_additional_clips()` 只接受已准备的本地 motion：它先写入并 SHA-256 登记每个 OBJ frame，之后才原子更新 production manifest，绝不生成合成帧或注册缺失文件。production manifest 的 `stand_up` 现直接注册为 `sit`（或 production 的 `sit_down`）的逆序帧；配饰派生 manifest 亦同。`stand_up` 的动作名和 `standing` marker 不变，但严格 manifest loader 会拒绝任何重新登记原始 stand-up OBJ 序列的 bundle；`MeshSequenceBackend` 仅为旧的已生成场景保留相同的兼容解析。2026-09-14 用户确认 CMU/BABEL sequence `3182` 的 `point to the right` 候选可用于本地 production 注册；它以 8 FPS、22 帧、`gesture_point_complete @ 0.9` 烘焙并登记在共享的 ignored production manifest 中。`OFFICE_CLIPS` 与 `MujocoNpcActionDriver` 因而允许 `STAND_UP`、`PUT_DOWN`、`GESTURE_WAVE` 和 `GESTURE_POINT`；point/wave 都是直接播放的完整 mesh sequence，不宣称 OBJ backend 具备尚未实现的上半身 overlay。

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

`office_scene.xml` 通过 `custom/text:npc_trajectory_profile=office_v1` 显式登记该契约。`MujocoServer` 启动时读取该登记，解析生成 wrapper 中的原始 `office_scene.xml` include，核对 SHA-256 并 preflight 全部路线；scene bytes、profile 或 anchor 漂移会使启动失败。`MujocoNpcActionDriver` 同时接收 production roster 的初始逻辑位置：当动作与 profile 的 source/destination/action 一一匹配时，它会在 `MOVE_TO` payload 写入 `trajectory_route`、`trajectory_source` 与 `route_mode=audited`；`NpcController` 会拒绝 route ID、source、目标 site 或动作许可不一致的命令。未匹配 profile 的可达动作仍是 `route_mode=dynamic`，它只表示当前碰撞 geometry 的实时规划，绝不表示已通过 profile preflight。

production roster 的十名 NPC 现在各自引用一个 `npc_spawn_*` scene site。scene builder 的回归测试要求十个初始 XY 坐标均不同、任意两者距离至少 0.5 m、生成 NPC 之间零初始接触。使用下列命令渲染多人总览；它不会隐藏任何 roster NPC，并为每个人写入 segmentation 可见像素审计：

```bash
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python tools/build_npc_scene.py \
  --population stretch_mujoco/models/office_population.production.example.json \
  --output stretch_mujoco/models/.population_overview.xml --include-base-scene
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python tools/render_npc_acceptance_video.py \
  --scene stretch_mujoco/models/.population_overview.xml \
  --population stretch_mujoco/models/office_population.production.example.json \
  --population-overview --output /tmp/npc_population_overview.mp4
```

preflight 还会用 `LocomotionController` 逐 10 ms 推进每条 route，包含每个渐进转向姿态；MuJoCo 必须报告 NPC collision proxy 与所有非 `office_floor` 场景 geom 零接触，否则以 `trajectory_route_collision` 拒绝。`office_v1` 当前分别审计了 336、315、554、674 个 move/turn pose，均为零接触；JSON receipt 对每条 route 写入采样数与 `collision_free: true`。运行时 planner 仅排除当前 NPC 自身，其他 mocap NPC collision proxy 会进入动态占用检查；检查以 250 ms 有界周期进行，下一段被动态占用时最多按该命令的 replan budget 重规划，否则终止为 `route_blocked_dynamic`，不会把穿越对方当作成功进度。chair navigation ingress 与 sit site 不再以写入 `mocap_pos` 跳转，最后一段同样按 `speed * dt` 连续推进。

## 20. 工位 work 会话（当前事实）

`OfficeAgentRuntime.submit_action(WORK)` 现在是一个 receipt-gated 的工位会话入口，而不是直接在 desk 位置播放 work。它把日程、LLM 和 API 的同一 public `WORK` 请求统一展开为 `MOVE_TO(chair) → SIT(chair) → WORK(workstation) → STAND_UP(chair) → IDLE`；只有 `SIT` 的物理回执成功并提交 `OCCUPIED_BY` 后，内部 work 命令才会被接受。work 结束后，`STAND_UP` 成功才释放椅子占用和 reservation，因此不会出现工作中站着、或失败时提前释放座位的状态。driver 还把 receipt-gated seated state 作为移动 gate：坐着的 NPC 不能直接 `MOVE_TO`；REST planner 也会排入 `STAND_UP → IDLE`。当前没有经批准的 REST、ATTEND_MEETING 或 OPEN_CABINET 实体 marker/asset，因此在启用 `MujocoNpcActionDriver` 时这些 public actions 以稳定错误拒绝，不能用逻辑计时冒充物理完成。

椅子不是由 `chair_right` 一类名字或坐标猜测：场景必须恰好声明一条 `Chair --NEAR--> Workstation` 关系。缺少或多条关联均会以稳定验证错误拒绝 work。实体桥接还从语义 interaction point 读取每个 `desk_work_site` 和 `chair_sit_site`；chair site 必须显式给出有限 `attributes.yaw`，否则 driver 在新场景装配期失败。因而迁移场景只需提供新的对象 ID、上述关系和 sites/yaw，不需要修改 office 专用动作代码。实体 `WORK` 在前序 sit 已确立的 chair site 播放并保持 `seated_idle`，不会再导航到 desk site 使 NPC 离开椅子。

可用以下命令生成实际 controller 驱动的路线巡回视频。四条 route 依次由真实 `MOVE_TO` 命令执行；路线之间的 source 切换明确标为静态 placement，不会伪造穿模移动。renderer 在开始写 MP4 前强制运行上述路径与转向碰撞审计，并把每条 route 的审计采样数写入画面 HUD 和 sidecar report：

```bash
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python \
  tools/render_npc_trajectory_tour.py \
  --scene stretch_mujoco/models/office_scene.xml \
  --profile stretch_mujoco/npc/trajectory_profiles/office_v1.json \
  --output /tmp/npc_all_routes.mp4
```

## 21. Control-sequence receipt lifecycle（当前事实）

`stretch_mujoco/agents/control_sequences/` 现提供 v1 YAML 的严格加载、静态能力预检、模式解析和计划段接口；`tools/validate_control_sequence.py` 只读取 YAML、scene XML、语义和 population 后编译，绝不启动 simulator。`--control-mode` 的唯一优先级仍是 CLI 显式值、YAML 默认值、再到 `yaml`；在 `llm` 模式中 YAML 的固定 steps 仅作策略/验收配置，不会被暗中执行，短计划必须经同一 `SequenceCompiler.compile_segment()` 后才能交给 executor。

实体对话不再把 driver 的 `RUNNING` 当作失败或成功：runtime 在提交 candidate 时只登记 pending turn 并标记 `PLAYING_TURN`，每次 `tick()` 轮询 interaction driver；只有成功的终态 physical receipt 才写 transcript、memory 和 `dialogue_turn_committed` event。approach/alignment/talk 的失败会走已有的统一 terminal cleanup，释放 participant reservation，且不提交文本。handover 与双人 conversation 的 role sites 由 driver 以 workflow/session ID lease；同一站位的第二个 workflow 稳定失败为 `interaction_sites_busy`，失败、取消、超时 receipt 路径释放 lease。带 `target_site` 的 ALIGN、关键动画和 handover 阶段还会持续检查当前位置；站位不存在或参与者被移动后，命令以 `unknown_target_site` 或 `target_site_not_reached` 失败，而不是凭旧 approach receipt 释放或接收物体。取消 command 亦有独立 receipt lifecycle：walk 必须等待安全 foot marker 时，target 与 cancel receipt 均可先为 `RUNNING`；target 最终 receipt 会原子投影到关联 cancel command，因而重放同一 cancel ID 必定得到其终态而不是永久的 `waiting_for_foot_marker`。该变化尚未使 renderer 成为 control-sequence runner，也未生成新的验收视频。

本切片实际验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_control_sequence_executor.py tests/test_conversation.py` 得到 `27 passed in 0.50s`；`PYTHONPATH=. .venv/bin/python tools/validate_control_sequence.py stretch_mujoco/models/control_sequences/npc_full_acceptance_v1.yaml --strict` 和同命令追加 `--control-mode llm` 都返回 `passed: true`。这些是静态/单元验证，不是 navmesh route preflight、MuJoCo integration 或 MP4 验收。

### 21.1 Stretch robot task transport（当前事实）

`StretchRobotTaskDriver` 是 `RobotTask` 的唯一真实 Stretch transport adapter：它只使用 live `StretchMujocoSimulator` 的 base `move_by`、status、`request_grasp_metrics`/`pull_grasp_metrics`、`attach_object_to_gripper` 和 `release_grasped_object` 路径。每一个 task 都必须经过 pickup waypoint base navigation、明确的 grasp IK acknowledgement、双指 contact observation、attachment、delivery navigation、release 和 delivery position observation，才会由 runtime 调用 `complete_robot_task(success=True)` 并写入 `stretch:<task>:delivered` receipt。`OfficeAgentRuntime.tick()` 会轮询该 driver；runtime 仍只拥有 receipt 后的 semantic commit。

当前 server public API 没有 semantic-site navigation 或 grasp IK command/receipt：`StretchMujocoSimulator.move_to()` 仅支持单 actuator 绝对位置，`move_by()` 的 base motion是相对量，且没有 `navigate_to_site`、`solve_grasp_ik` 或 `is_grasp_ik_complete`。因此 composed scene owner 必须显式传入 world-coordinate waypoints，并提供 IK extension；任一能力缺失会写 terminal failed receipt（例如 `robot_waypoint_transport_unsupported` 或 `robot_ik_transport_unsupported`），绝不由 `MockRobotExecutor` 或 `complete_robot_task` 逻辑调用冒充成功。`RendererSimulator` 只有 NPC `NpcSystem` transport，默认同样得到这个明确失败 receipt；它不能渲染真实 Stretch delivery，直到 renderer 运行在包含 Stretch server transport、NPC population 和具名 waypoints 的同一 composed scene。

本轮验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_stretch_robot_task_driver.py tests/test_control_sequence_executor.py tests/test_conversation.py` 得到 `31 passed in 0.55s`。测试包含真实 transport 方法调用的 navigation → IK → bilateral-contact → attach → release → observation success chain，以及缺少 IK 时的 terminal failed receipt；尚未启动独立 MuJoCo server 或产出 MP4。

## 22. 声明式 NPC 场景组合（当前事实）

`stretch_mujoco.npc.composition.compose_npc_scene(population, output)` 是正式的
population-to-scene 构建入口。population JSON 中的 `scene`、`asset_manifest`、NPC
identity/appearance/spawn/capability 是唯一配置来源；API/CLI 不接受第二个 scene 参数。
它调用兼容的 `build_npc_scene(..., include_base_scene=True)`，在不修改 base office XML
的前提下生成可直接 `MjModel.from_xml_path()` 加载的 combined MJCF。为了让输出可放在
`outputs/` 或 `/tmp`，它仅生成相邻的 base/stretch include wrapper，并把嵌套 Stretch
assetdir 固定为 base scene 的绝对投影；wrapper 是可删除的构建产物，不是源场景修改。

composition receipt（默认 `*.composition.json`）钉定 population、base scene、manifest
和 generated scene 的 SHA-256，并记录 canonical NPC IDs、`npc__<id>` body、handover
site 与 spawn site。构建/缓存复用前后都会实际编译 MuJoCo 并验证这些名称；缺失项以
稳定 `composed_scene_missing_*` 错误失败。`load_composed_npc_runtime()` 用同一
population 共同构造 model、`NpcSystem`、semantic world 和 `OfficeAgentRuntime`，并把
base office 里的 legacy preview binding 映射为 canonical body/site，避免旧录制 clone
链与 population runtime 发生 unknown-NPC/semantic-binding 冲突。`native_scene.py` 仍只
是 recording fixture。MuJoCo 不能在已编译 model 热插 NPC；人数或外观变化必须更新
population 后重新组合并 reload。

### 22.1 Population 启动与配置边界（当前事实）

`launch_sim --population <population.json> [--semantics <world.json>]` 现在在系统临时目录
组合/复用 MJCF，并把同一 population path 传给 server；server 因而使用
`NpcSystem.from_population()` 和 manifest graph，而非对 composed model 的 `from_model()`
猜测。schema-v2 的 `population.trajectory_profile` 是该系统与 action driver 的唯一 profile
来源；XML custom text 仅保留给 legacy `from_model`。schema-v2 显式 `agent_id` 必须等于
`npc_id`；capability 会在 `OfficeAgentRuntime.validate_action()` 拒绝未授权的 locomotion,
sit, conversation, object handover 和 computer actions，idle/stand-up 不受此 gate。全局
`dialogue_policy` 可内联到 population，NPC profile 的 `sociability` 初始化实际 social
energy，故影响对话 admission。新的 interaction-point `attributes.binding` 可声明
location、seat ingress、placement、object approach 或 robot request 的 site/yaw，覆盖旧 office
常量；未迁移 schema-v1 场景仍使用明确的 legacy constants。

## 23. Runtime closure（当前事实）

`OfficeAgentRuntime` 现在为一次已验证的动作完成分配一个 event ID，并把同一 ID 写到
`action_succeeded`、`semantic_commit` 及相应 memory entry；事件类型仍分开，以保留物理
完成和语义投影的查询含义。没有 receipt 的 embodied action 不得借此路径完成：启用
`MujocoNpcActionDriver` 时，未注册 marker/asset 的 REST、ATTEND_MEETING、OPEN_CABINET
继续以稳定错误拒绝。

utility 的运行时约束只有 `OfficeAgentRuntime._utility_context()` 一个投影入口：日程紧迫度、
meeting priority、未决邀请、连续失败、对象 reservation、social cooldown 和近期重复会被
传入 `EmployeePlanner.choose_plan(..., context=...)`；utility 模块只负责对该值输入评分。进入
conversation 前 planner 将 pending plan 作为 checkpoint 并清空 queue，terminal cleanup 重新
验证 target 后恢复；若对象已不存在则清除 stale plan 并写 `plan_resume_invalidated` event。
这不取消 busy embodied action，仍遵循既有 receipt-safe 的会话 admission contract。

demo receipt 是派生产物而非可编辑配置。当前 checkout 的 new-demo 01–06、07 Scene A 与
07 Scene B 的 sequence/population/semantic/profile hashes 均与各自当前输入一致；07 的最终
montage 尚未由这两条当前 source render 重建，仍只能作为历史组合产物。动画
crossfade/point/gaze/overlay 与真实 3 NPC + Stretch 同时端到端集成仍是明确搁置项，不在本轮
完成声明内。后者的 P0 验收仅保留为本地失败复现，不能作为本分支的通过测试或 PR 交付证据：
当前真实 Stretch transport 缺少语义 waypoint、grasp-IK 与 release 验证闭环，且拥挤三 NPC
handover 的动态 rendezvous route 仍可能终止为 `route_blocked_dynamic`/`route_unavailable`。

本轮 PR 聚焦回归覆盖 NPC controller/asset/schema/composition、Agent action/
conversation/control-sequence、语义绑定、对象附着、导航和 Stretch robot-task
接口，命令为 `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
-q <27 focused test files>`，结果 `211 passed in 99.24s`。
`tools/validate_npc_assets.py --population
stretch_mujoco/models/office_population.production.example.json` 返回
`Validated 10 NPC(s) and 1 bundle(s)`；版本库中四份 control-sequence YAML 均经
`tools/validate_control_sequence.py <yaml> --strict` 验证为 `passed: true`。核心模块
`compileall` 和 `git diff --check` 通过。以上不代表完整 `tests/` suite 已在本轮运行。

## 24. 办公室/家庭统一语义场景编译（当前事实）

`stretch_mujoco.npc.scene_config`、`scene_compiler` 和
`semantics.scene_discovery` 已提供统一的严格 JSON `scene_npc_config/v1` 路径。编译器同时
发现 manifest asset/zone 和 XML semantic/action site，反向验证 body、geom、site 绑定，按
office/home policy 与家庭 exact-name taxonomy 分类，并生成 semantic v2、保守的 runtime v1
投影、带语义 site 的可移植 MJCF、population、trajectory、coverage 和 SHA-256 receipt。配置、
taxonomy、家庭实例豁免、办公室可变人口计划、家庭 room/slot provenance 均保存在
`stretch_mujoco/models` 的对应资源目录；临时目录只用于原子 staging，不作为事实源。

点位生成使用与 runtime NPC torso capsule 一致的 `agent_radius=0.16`。办公室采用
`clearance=0.06`、`resolution=0.08`；家庭窄门场景采用 `clearance=0`、`resolution=0.06`，
避免栅格离散和额外膨胀把实际可通过的门洞误切为孤岛。自动点必须位于主可行走连通分量。
桌面 graspable 复用支撑家具 approach；seat 同时生成可达
approach 和位于资产顶面的 sit action point；observation 排除 render-only geom 并要求碰撞组
视线。家庭旧 demo slot 会确定性投影到主分量内的互异位置，并把最终坐标写入逐场景配置；
bathroom、bedroom、kitchen、living-room、office 以 asset-hint medoid 生成逐场景 room seed
overlay，并在 provenance 中记录 anchor。办公室人口由 `office_population_plans.json` 配置为
2–4 人，家庭人口由逐户 roster 配置；两类场景都有差异化业务路线，全部 NPC 到全部 required
点的覆盖路线由编译器自动生成。`home_scene_overrides.json` 保留实例修正入口；当前 20 个
active 场景不需要语义豁免。

`tools/audit_active_scene_semantics.py --compile-output-root` 先在隐藏 staging 中编译所有场景，
全部成功后才以内容寻址 build 和原子 `active_catalog.json` 发布。当前 active build ID 为
`6ecc0ea9c8d86957031ed5ebac761c8d52a31fac9d039d54b48462a6e30896ff`：20/20 场景、1310 个
语义实体、1350 个 required navigation point、62 个 NPC、4100 条业务/覆盖路线，所有正式
coverage 均为 `unresolved=0`、`unbound=0`、`unreachable=[]` 且 `strongly_connected=true`。
相同输入连续两次全量构建得到同一 build ID；所有输出 receipt hash、population、semantic
v1 和 trajectory loader 均已复核，办公室 02 与家庭 06 的生成 MJCF/runtime trajectory
preflight 实际通过。

运行时导航已与上述编译契约统一：`NpcSystem.from_population()` 和带 profile 的
`NpcSystem.from_model()` 都把 trajectory profile 的 surface、NPC 半径、clearance、grid
resolution 与排除 body roots 注入每个 `LocomotionController`。所以办公室使用
`office_floor/0.16/0.06/0.08`，家庭使用
`hssd_floor_collision/0.16/0/0.06`，首次规划、超时重规划和动态 NPC 占用检查均在同一份
碰撞几何上执行。profile 已绑定却缺少指定 surface 时稳定失败为
`navigation_surface_missing:<surface>`，route 无法生成时失败为 `route_unavailable`；生产
场景不存在直线 fallback。旧 fallback 只保留给没有 profile 的最小协议 fixture。

聚焦回归命令为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
tests/test_generated_home_npc_preprocessing.py tests/test_generated_office_npc_preprocessing.py
tests/test_npc_composition.py tests/test_scene_npc_compiler.py tests/test_npc_trajectory_profile.py`，结果
`42 passed in 414.64s`。本轮没有执行会物化大型 DEM/DOM 资产的 Resolve。
### Active NPC showcase foundation (partial evidence)

The version-controlled scenarios are under `aaa_workspace/showcases/`, with the shared runner at
`tools/render_active_scene_showcase.py`. Office and home movement uses the active trajectory
profile and emits the existing route/preflight/continuous collision receipts. The runner currently
executes only one selected movement phase; all other phases are marked `not_executed`, so these
outputs are partial evidence rather than complete demos. Conversation and robot handover status
is read from the generated population capability projection; unsupported source declarations keep
their explicit fail-safe reason.
Generated active artifacts are not edited by the showcase runner.

All 20 office/home source scene configs now carry the same strict `traffic` policy and explicit
`conversation`/`robot_handover` capability outcomes. The compiler projects these into each
population as `traffic_policy` and `interaction_capabilities`; the showcase runner reads those
projections and reports the configured reason for unsupported phases. No source scene currently
claims supported interaction without source-backed sites, object binding, and preflight.

## 25. Three-NPC asynchronous LLM/MuJoCo benchmark (verified 2026-09-17)

`examples/llm_multiagent_mujoco_benchmark.py` runs a reproducible accelerated workday with three NPCs
in the active office scene `office_01_linear_bench`. It creates a temporary, model-validated benchmark
fixture from the active office assets; the active source assets are not modified. Each NPC has an
independent deterministic LLM session, while `Transport.step()` continues advancing MuJoCo during
provider waits. The scenario covers work, drink, rest (`sit -> idle -> stand_up`), and dialogue.

Verified command:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python examples/llm_multiagent_mujoco_benchmark.py --output aaa_workspace/experiments/llm_multiagent_benchmark/report.json --provider-latency 0.01`

The report passed with 3 NPCs/3 sessions, 8 physics steps during LLM wait, workday clock 09:00 to
18:00, replan and safe-idle recovery, 0 collision contacts, minimum separation `0.580 m` against
the policy threshold `0.44 m`, conversation receipts `approach=2`, `alignment=2`, `talk=1`, and
semantic orphan count `0`. The JSON report is at
`aaa_workspace/experiments/llm_multiagent_benchmark/report.json`.

The active source population declares conversation unsupported, so this benchmark uses a temporary
role-site fixture and reports it as `benchmark_fixture`; this does not claim that production
population conversation capability is enabled. The provider is an injectable asynchronous test
adapter; a real LLM API adapter can replace it without changing the physics/receipt contract.

The same runner now accepts `--video-output`, `--video-fps`, `--video-width`, and
`--video-height`. Video mode renders the live composed MuJoCo state with a fixed global top-down
camera and a three-column subtitle panel for LLM request/result/recovery events, every physics
advance, and terminal action receipts. The verified H.264 artifact is
`aaa_workspace/experiments/llm_multiagent_benchmark/three_npc_global_topdown.mp4`; its matching
report is `video_report.json`. The verified file is 1280x720 at 10 FPS, contains 731 frames
(73.1 seconds), ends at workday clock 18:00, and the benchmark remains `passed: true`.

## 26. Three-NPC active-home LLM/MuJoCo video demo (verified 2026-09-17)

`examples/llm_multiagent_home_mujoco_demo.py` runs three independent LLM sessions in active scene
`home_04_103997970_171031287`. It creates a temporary population projection only to start the
runtime clock at 18:00; active scene, trajectory, population, and semantic artifacts are not
modified. The demo follows the declared sequential route-reservation policy and executes three
compile-time-audited routes: Alex to the kitchen, Jordan to the living room, and Morgan to the
bedroom. One injected Morgan provider failure is recovered by replan while physics continues.

The home runtime semantic projection exposes the coarse action location `zone.home`; exact room
destinations remain owned by the trajectory profile and are recorded separately in the report.
Generated home furniture includes seatable classifications without complete `chair_sit_site`
bindings, so this demo deliberately limits its action driver to explicit MOVE_TO route/site
mappings. It does not bypass the seat validation contract or claim conversation/robot handover,
both of which remain `unsupported` in the active population.

The verified H.264 artifact is
`aaa_workspace/experiments/llm_multiagent_home_demo/three_npc_home_global_topdown.mp4`, with matching
`report.json`. It is 1280x720 at 10 FPS, contains 269 frames (26.9 seconds), uses a fixed global
top-down camera, and displays LLM, physics, and terminal action-receipt timelines. The report is
`passed: true`, records three isolated sessions, 4 physics steps during LLM waits, no NPC collision,
and minimum NPC separation `0.767 m` against the home policy threshold `0.32 m`. The shared office
benchmark regression remains green: `1 passed in 86.79s` for
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_llm_multiagent_mujoco.py --tb=short`.

## 27. Interaction site plan schema Phase 1A（2026-09-17，candidate only）

`stretch_mujoco.npc.interaction_station` 新增严格的
`interaction_site_plan/v1` 权威源 loader/model。该模块只拥有 authored interaction facts、
稳定 site-name 派生和 source manifest/MJCF cross-reference；它不拥有导航、碰撞、动作执行或
runtime capability/evidence 判定。plan schema 不包含 `validated_for_runtime`；该字段只能由
Phase 1B compiler receipt 拥有。loader 拒绝未知/缺失字段、非 array root collection、重复
JSON key、NaN/Inf、非法 yaw、重复
resource ID、同位或不相向的角色、距离契约违规、未绑定或类别错误的 seat/workstation/computer、
非 dynamic/graspable handover object、未绑定 region、缺失 seat ingress 和未绑定 workstation
seat slot。MJCF
body inventory 会递归读取本地 include，因此 manifest 中的 `base_link` 必须在实际 included
robot XML 中存在。

`stretch_mujoco/models/scene_interaction_plans/office/office_02_cross_axis.json` 是首个 candidate
权威 plan，包含 1 个 conversation station、1 个 handover station、2 个独立 seat slot 和
1 个 workstation binding。它复用 source manifest/MJCF 中的 chair、workstation、iMac、
graspable bottle 和 Stretch body 身份，并投影 `PART_OF`、`Computer ON Workstation` 与
`SeatSlot NEAR Workstation` 关系。handover 按 declared mode 校验同时参与的角色对；candidate
当前声明 `npc_to_npc` 和 `robot_to_npc`，不预先宣称尚未布局验证的 `npc_to_robot`。alternative
giver/robot role 可以共享位置，但每个 declared mode 中同时出现的 pair 都必须满足距离和相向约束。

本切片没有修改 `scene_npc_config/v1`、population v2、compiler、runtime、active catalog 或
generated artifact，也没有改变 20 个生产场景的 capability status。office_02 candidate 的坐标
和绑定仅通过 strict schema/source identity 检查，不构成 NPC/robot navigation、静态碰撞、
work pose、sit marker 或 handover workspace 的物理证据。unreachable pose 和 robot route
failure 必须等 Phase 1B compiler/preflight 使用 MuJoCo navigation mesh 后再做 planted negative；
Phase 1B receipt 必须写 `validated_for_runtime: false`，直到物理验收闭包。本节不得被引用为
conversation、handover、sit 或 work 的 runtime support 声明。

Phase 0 基线：

- `python -m py_compile` 对 scene config/compiler/schema/world/scene discovery/drivers/
  interactions/runtime/robot handover 通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... pytest tests/test_interactions.py
  tests/test_npc_runtime_safety.py`：`6 passed`。
- compiler/semantic/action 聚焦集合：`50 passed, 1 failed`；既有失败为
  `test_production_driver_requires_scene_binding_for_registered_gesture_point`，实际集合比旧断言多
  `ActionType.IDLE`。Phase 1A 未修复或隐藏该无关 mismatch。
- `tests/test_robot_handover.py` 当前不存在。

Phase 1A 验证：

- `.venv/bin/python -m py_compile stretch_mujoco/npc/interaction_station.py
  tests/test_interaction_station_schema.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_interaction_station_schema.py --tb=short`：`19 passed in 0.52s`。
- 同一环境运行 `tests/test_interaction_station_schema.py tests/test_scene_npc_compiler.py
  --tb=short`：`32 passed in 1.63s`；旧 v1 compiler 路径未受 Phase 1A schema 修正影响。

## 28. Scene config v2 与 population v3 Phase 1B-1（2026-09-17）

`stretch_mujoco.npc.scene_config` 现可严格读取 `scene_npc_config/v1` 和
`scene_npc_config/v2`。v2 用 `SceneInteractionPlanRef` 引用唯一 authored
`interaction_site_plan/v1`，并通过 `required_capabilities` 声明 compiler 必须验证的
conversation/handover/sit/work 范围；`RobotNavigationConfig` 独立声明 Stretch footprint、
clearance 和 resolution。v2 root 不接受旧 `interactions`，plan ref 也不接受 `status` 或
`sites`，因此作者不能用配置文字绕过 compiler receipt。v1 loader 的既有 interaction
status/sites 读取和错误语义保持兼容，但其 `interaction_plan`/`robot_navigation` 始终为空，不能
被解释为 multi-station evidence。v2 在 compiler 接入前只保守投影内部 unsupported declaration，
不产生 production support。

`stretch_mujoco.npc.schema` 新增 population schema v3 parser/model。v3 以完整
`interaction_stations` catalog 表示 conversation、handover、seat 和 workstation；每项严格校验
role 集、site/yaw、actor pairs、handover modes、object IDs 和字段集合。conversation/handover
还必须保留严格的 `distance_m` 和 `yaw_tolerance_rad`，handover 额外必须声明
`transfer_site`，seat 额外必须声明 `slot_index` 和 `clearance_radius_m`。调用方提供
MuJoCo site universe 时，parser 验证每个 role site 和 transfer site 均存在。station ID 在全部
kind 间必须唯一；workstation 的
`seat_slot` 必须引用同一 catalog 中的 seat station。v3 root 使用明确 required/optional 字段集合，
拒绝所有未知字段并单独拒绝旧 `interaction_templates`。schema v2 root 行为保持宽松兼容，且不解析
或存储 v3-only 的严格 `spawn_policy`；已有 template 只投影为带
`legacy_single_station: true`、`production_evidence: false` 的 adapter，不能成为生产支持证据。

本 checkpoint 尚未修改 scene compiler 或 `SemanticWorld`，没有生成 v3 population/MJCF/receipt，
没有执行 NPC/robot route preflight，也没有修改 generated artifacts、active catalog 或 20 个生产
config capability status。这些属于后续 Phase 1B compiler checkpoint。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/npc/scene_config.py
  stretch_mujoco/npc/schema.py tests/test_scene_contract_v2.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_scene_contract_v2.py --tb=short`：`17 passed in 0.48s`。
- 同一环境运行 `tests/test_scene_contract_v2.py tests/test_scene_npc_compiler.py
  tests/test_semantic_action_bindings.py tests/test_npc_assets.py --tb=short`：
  `42 passed in 63.43s`。

## 29. SemanticWorld v2 role/relation fidelity Phase 1B-2A（2026-09-17）

`SemanticWorld._from_v2_payload()` 现按 point payload 的 `role` 构造
`InteractionPoint`，不再把全部 v2 point 降为 `human_stand_site`。新增且严格解析的角色值为
`conversation_speaker_site`、`conversation_listener_site`、`handover_giver_site`、
`handover_receiver_site`、`handover_robot_site`、`handover_transfer_site` 和 `seat_ingress_site`；seat sit 与 desk work
继续使用既有兼容字符串 `chair_sit_site` 和 `desk_work_site`。原有
`conversation_site`、`handover_site` 等枚举值未改名。

v2 loader 新增 `ObjectType.SEAT_SLOT`（`resource.seat_slot`）和
`RelationType.PART_OF`，并将 payload `relations` 解析为 `SemanticRelation`。关系端点继续由
`SemanticWorld._validate_graph()` 验证；未知 role、未知 relation、未注册 subject/object 均
fail closed。point payload 的 `station_id`、`slot_id`、`binding`、`target`、`yaw` 及其他字段
保留在 `InteractionPoint.attributes`。无 XML binding 的 region 仍作为明确
`topology_only` object 加载，保持既有 v2 region 行为。legacy v1 parser 未修改。

本 checkpoint 只提高 loader fidelity；scene compiler 尚未生成上述角色、seat slot entity 或
relations，也未产生 route/collision/runtime evidence。没有修改 compiler、runtime、generated
artifacts、active catalog 或生产 capability status。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/semantics/world.py
  tests/test_semantic_world_v2.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_semantic_world_v2.py --tb=short`：`6 passed in 0.43s`。
- 同一环境运行 `tests/test_semantic_world_v2.py tests/test_semantic_action_bindings.py
  tests/test_scene_npc_compiler.py tests/test_scene_contract_v2.py --tb=short`：
  `39 passed in 1.57s`。

## 30. Interaction plan 纯投影 Phase 1B-2B1（2026-09-17）

`stretch_mujoco.npc.interaction_projection.project_interaction_plan()` 现将已通过严格 loader 的
`SceneInteractionPlan` 确定性投影为候选 derived-site descriptor、semantic entity/point/relation
fragment、population v3 `interaction_stations` 以及 candidate receipt 骨架。纯投影函数本身不写文件；
后续 Phase 1B-2B2a 已将其 MJCF/population 子集接入 `compile_scene_npc_config()`，见下节。

投影包含所有 conversation speaker/listener、handover giver/receiver/robot/transfer、seat
ingress/sit 和 workstation work site。handover transfer 是专用透明 site 和
`handover_transfer_site` semantic point，不伪装成 actor role。catalog 保留 plan 中的 distance
range、yaw tolerance、transfer site、seat slot index 和 clearance radius。跨不同 station ID 的完全
相同 coordinate+yaw 会以 `duplicate_physical_station_pose` fail closed；同一 handover station 内不同
mode 不同时使用的重合 giver/robot 仍按 authored contract 保留。

office_02 候选 plan 投影 11 个 interaction site，其中包含 1 个 transfer site。测试中将第二组
conversation/handover 的所有 role position 和 transfer position 显式偏移后，投影完整产生 17 个
site，不存在 `[0:2]` 截断。candidate receipt 仅可记录 `plan_schema` 与派生 ID 唯一性；
`source_bindings`、NPC/robot navigation、terminal collision、handover reach、sit action 和 work action
全部是 `not_run`，`validated_for_runtime` 固定为 `false`。这些 fragment 不是生产运行支持证据。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/npc/interaction_projection.py
  stretch_mujoco/npc/schema.py stretch_mujoco/semantics/world.py
  tests/test_interaction_projection.py tests/test_scene_contract_v2.py
  tests/test_semantic_world_v2.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_interaction_projection.py tests/test_scene_contract_v2.py
  tests/test_semantic_world_v2.py --tb=short`：`35 passed in 0.56s`。
- 同一环境运行 `tests/test_interaction_projection.py tests/test_interaction_station_schema.py
  tests/test_scene_contract_v2.py tests/test_semantic_world_v2.py
  tests/test_semantic_action_bindings.py tests/test_scene_npc_compiler.py --tb=short`：
  `73 passed in 1.81s`。

## 31. v2 MJCF/population 原子写出 Phase 1B-2B2a（2026-09-17）

`compile_scene_npc_config()` 现在加载 config 后仅按 `config.schema` 分流；v1 分支相对
`aaa_workspace/archive/20260917_repository_cleanup/stretch_mujoco/npc/scene_compiler.py.bak-20260917_103207`
除四行 v2 dispatch 外字节不变。v2 分支由
`stretch_mujoco.npc.scene_compiler_v2` 负责，使用 config 的 scene ID、source manifest 和 source
MJCF 重新调用 `load_interaction_site_plan()`，然后调用唯一的
`project_interaction_plan()`。

2B2a 仅在 output 目录的 temporary staging directory 内组装候选 MJCF 和 population，所有
schema/source/projection/population 校验成功后才用 `os.replace()` 替换最终文件。MJCF 包含既有基础
semantic sites 和全部 11 个 office_02 interaction sites，派生 site 均为 `rgba="0 0 0 0"`
透明站点。population 为 schema v3，包含完整 `interaction_stations`，并保留选中 NPC 的
profile/embodiment/needs/schedule/capabilities、clock、asset/appearance/trajectory references、traffic 和
spawn policy。资源路径按最终 output 目录重写，写出前使用 `NpcPopulation.from_dict()` 做 v3
contract 自检。

本 checkpoint 当时故意不写 semantic interaction additions、coverage、trajectory projection 或 receipt；
该输出限制已被后续 Phase 1B-2B2b 部分取代，见下节。2B2a 和当前 2B2b 都不执行
NPC/robot navigation、collision、reach、sit/work action preflight。population 的
`interaction_capabilities.production_evidence` 为 `false`；这两个输出不能被解释为 runtime
validation 或 production support。输出只写入测试 `tmp_path`，未修改 active/generated tree。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/npc/scene_compiler.py
  stretch_mujoco/npc/scene_compiler_v2.py tests/test_scene_compiler_v2.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_scene_compiler_v2.py --tb=short`：`3 passed in 0.68s`。
- 同一环境运行 `tests/test_scene_compiler_v2.py tests/test_scene_npc_compiler.py
  tests/test_interaction_projection.py tests/test_scene_contract_v2.py --tb=short`：
  `45 passed in 1.83s`。

## 32. v2 semantic/receipt 候选写出 Phase 1B-2B2b（2026-09-17）

`stretch_mujoco.npc.scene_compiler_v2` 现在保持 2B2a 的私有 backend 边界，公开入口仍只有
`compile_scene_npc_config()`。v2 编译先以 config scene ID/source manifest/source MJCF 重新加载
plan，并 fail closed 校验 `required_capabilities` 中的 conversation/handover/sit/work 都有对应
plan resource。

compiler 通过 base semantic entity 的 `source.id` 和 `source.xml_binding.name` 将 plan 中的
XML-body alias 唯一解析为 canonical semantic ID。新 seat-slot entity、11 个 interaction point 和
`PART_OF`/`ON`/`NEAR` relation 会合并到 `compile_semantics()` 的 v2 输出；workstation/computer
仅做 targeted update。递归 merge 保留既有 source binding、labels、navigation/action points 及其他
discovery metadata，并将 work point 追加到 workstation `points.action`。new entity/point ID 冲突、
unbound/ambiguous alias、重复或同 subject+relation 指向不同 object 的 relation 均 fail closed。写出的
semantic v2 在替换最终文件前必须通过 `SemanticWorld.from_json()`。

v2 现与 v1 保持相同的七类输出键：scene、semantic v2、semantic v1、coverage、population、
trajectory profile 和 receipt。semantic v1 仅保留 base compatibility projection，明确写入
`projection_scope: base_compatibility_only` 和 `interaction_station_support: false`，不投影新 station
role。coverage 的 `navigation_preflight` 为 `not_run`；trajectory 仅是可被
`NpcTrajectoryProfile` 解析的声明式 route contract，未执行 preflight。

candidate receipt 组合纯投影 validator state，并记录 config、plan、source MJCF、source manifest、
semantic policy 和 NPC catalog 六个输入 SHA-256，以及 receipt 之外全部六个输出文件的
SHA-256。receipt 不自哈希，不包含 timestamp 或 temporary path，同一 config/output path 两次编译的
所有输出字节一致。只有 compiler 完成 plan loader、source alias、semantic graph 和 population/profile
contract 检查后，top-level `source_bindings` 才为 `passed`；嵌套的纯 projection state 仍是
`not_run`。`npc_navigation`、`robot_navigation`、`terminal_collision`、`handover_reach`、
`sit_action` 和 `work_action` 全部为 `not_run`，`validated_for_runtime` 为 `false`。

测试使用 `tmp_path` 构造非 active office_02 v2 config，逐一复算全部 input/output hash，并植入
required capability 缺失、semantic entity ID 冲突、relation 冲突和 printer-as-computer。编译前后
active catalog SHA-256 保持为
`bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365`。生产 capability
status 未修改，此 checkpoint 仍不是 runtime support 证据。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/npc/scene_compiler.py
  stretch_mujoco/npc/scene_compiler_v2.py stretch_mujoco/npc/interaction_projection.py
  tests/test_scene_compiler_v2.py tests/test_interaction_projection.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_scene_compiler_v2.py tests/test_interaction_projection.py --tb=short`：
  `13 passed in 0.81s`。
- 同一环境运行 `tests/test_scene_compiler_v2.py tests/test_scene_npc_compiler.py
  tests/test_interaction_projection.py tests/test_interaction_station_schema.py
  tests/test_scene_contract_v2.py tests/test_semantic_world_v2.py
  tests/test_semantic_action_bindings.py --tb=short`：`81 passed in 2.05s`。

## 33. v2 NPC/robot navigation preflight Phase 1B-2C（2026-09-17）

`stretch_mujoco.npc.interaction_station` 在 source identity 校验之外新增三项 authored
semantic-spatial integrity guard。manifest zone/region 提供 bounds 时，conversation 的每个 role
以及 handover 的每个 role 和 transfer XY 必须位于所声明区域内，否则稳定失败为
`station_pose_outside_region`。seat ingress 到 sit pose 的水平距离必须严格大于该 slot 的
`clearance_radius_m` 且不超过成人接近上限 `1.25m`，否则为
`seat_ingress_distance_invalid`。computer 有显式 `support` 时必须解析到同一 workstation；没有
显式 support 时，manifest 必须提供可用的 workstation/computer XY 和非负 workstation
collision radius；任一证据缺失或不可用均稳定失败为
`workstation_computer_support_unverifiable`。证据完整时，computer 必须位于该 footprint 加
`0.25m` 容差内，否则为 `workstation_computer_support_mismatch`。缺 workstation position、缺
computer position、缺 radius 和 measurable mismatch 均有 planted negative；compiler 不会把证据
不足或不一致的 authored fact 自动移动、补值或重绑。显式 `support` 的严格解析路径保持不变。

`stretch_mujoco.npc.scene_compiler_v2` 现将 public
`compile_scene_npc_config(..., check_navigation=...)` 的开关同时传给基础
`compile_semantics()` 和 interaction preflight。`True` 使用基础 compiler 已有的物理投影点和
coverage evidence；`False` 保留基础 `navigation_preflight: not_run`，并将 NPC/robot interaction
validator 明确写为 `not_run / check_navigation_false / candidate_only`，不能成为 runtime evidence。

interaction preflight 从实际 staging MJCF 创建两个 `OfficeNavigationMesh`，两者使用同一配置
surface，且都排除 manifest 声明的 Stretch root body。NPC mesh 使用
`agent_radius=0.16 + clearance=0.06`，robot mesh 使用独立的
`footprint_radius=0.32 + clearance=0.08`；二者 resolution 均为 `0.08`。每个目标必须是主连通域
中的 free cell，之后才规划路线，不会对 authored interaction target 做 nearest snapping。NPC 从
每个配置 population spawn 到 conversation speaker/listener、mode 对应的 handover giver/receiver、
每个 seat ingress 规划；workstation 复用其 seat ingress。相同 ingress 的 sit/workstation usage
只生成一个物理目标和一组路线，但 receipt 在该 target 的 `usages`/`consumers` 中保留两种用途。
robot 从 source MJCF 观测到的 `base_link` XY 到 robot_npc conversation 的两个可变角色点及
robot handover role 规划。receipt 保存 footprint、主连通域、origin、target、site 和路径长度。

office_02 candidate plan 已根据上述真实 mesh、manifest bounds 和 owner geometry 做
navigation-only 校准。meeting conversation speaker/listener 分别为
`[-7.26, 1.82, 0.025] / yaw=-1.57079633` 和
`[-8.06, 1.82, 0.025] / yaw=1.57079633`，间距 `0.8m`，均在
`zone.meeting=[-9,-2]x[0,6]` 内并同时属于 NPC/robot 主连通域。handover 保留已测量的 work-zone
位置：giver/robot `[-0.36,-4.88,0.025]`、receiver `[0.84,-4.88,0.025]`、transfer
`[0.24,-4.88,0.92]`。

seat pilot 使用 manifest meeting chair 013 和 011。chair 013 的 sit/ingress 为
`[-7.05,3.00,0.46] / [-7.98,3.02,0.025]`，水平距离 `0.930215m`；chair 011 为
`[-4.725,4.73205081,0.46] / [-4.38,5.58,0.025]`，水平距离 `0.915447m`。两者都满足
`0.28m < distance <= 1.25m` 并在 NPC 主连通域。chair 010 的最近 NPC-free 主连通域 cell 为
`[-3.90,5.18]`，距 owner/sit `2.180573m`，因此明确排除出本 pilot，没有使用远程 ingress。

workstation 绑定改为 manifest instance 8 / XML body `asset_008_new_team` / authored alias
`object.asset_008_new_team` / canonical semantic ID `object.new_teaming_table.8`，computer 为
`object.asset_009_new_imac`，两者 manifest center XY 都是 `[-5.5,3.0]`，水平支撑距离为 `0m`。
workstation 使用 `seat.object.asset_013_new_dini.01`，work pose 为
`[-7.05,3.0,0.73] / yaw=1.57079633`。yaw、station distance/facing contract 均未放宽。基础 compiler
将四个原始 population spawn 物理投影为 `[0.02,-1.30]`、`[-6.70,3.82]`、
`[1.25,3.00]`、`[6.74,3.74]`。

当前成功 receipt 为 4 个 NPC、6 个唯一 NPC target、24 条 NPC route，以及 3 个 robot target、
3 条 robot route。导航 planted negatives 分别稳定失败为
`npc_navigation_target_unavailable:point.conversation.meeting.01.speaker` 和
`robot_navigation_mesh_unavailable`；前者是 meeting bounds 内但落在 meeting-table 膨胀障碍中的
合法相向 pair，后者只把 schema-valid robot footprint 扩大到 `5.5`，且同次 preflight 在创建
robot mesh 前已经完成 NPC 24 条路线。输出 MJCF 由 MuJoCo 实际加载，而不只是 XML parse。

这些坐标仅为 navigation-validated candidate。`terminal_collision`、`handover_reach`、
`sit_action`、`work_action` 仍为 `not_run`，sit/work 高度与动作 marker 尚未物理验收，receipt 的
`validated_for_runtime` 继续为 `false`。没有发布 v2 candidate、修改 active catalog 或启用生产
capability；20 个 office/home source config 仍保留 40 条 `status: unsupported`。active catalog
SHA-256 仍为
`bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365`。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/npc/scene_compiler.py
  stretch_mujoco/npc/scene_compiler_v2.py stretch_mujoco/npc/interaction_station.py
  tests/test_interaction_station_schema.py tests/test_interaction_projection.py
  tests/test_scene_compiler_v2.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_interaction_station_schema.py tests/test_interaction_projection.py
  tests/test_scene_contract_v2.py tests/test_semantic_world_v2.py
  tests/test_semantic_action_bindings.py tests/test_scene_npc_compiler.py
  --tb=short`：`76 passed in 1.74s`。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_scene_compiler_v2.py --tb=short`：`12 passed in 101.70s`。
- Final Phase 1 correction：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_interaction_station_schema.py --tb=short`：`28 passed in 0.59s`；同轮
  `.venv/bin/python -m py_compile stretch_mujoco/npc/interaction_station.py
  tests/test_interaction_station_schema.py` 与 `git diff --check` 通过。该 correction 未修改
  compiler 或 plan，因此未重复运行约 102 秒的 compiler v2 suite。

## Phase 2A interaction station allocator（未接入 runtime）

`stretch_mujoco.agents.interaction_stations` 现提供 population v3
`interaction_stations` 的严格不可变 catalog adapter 和基于 `RLock` 的内存 allocator。
conversation 支持 `npc_npc`/`robot_npc`，handover 支持
`npc_to_npc`/`robot_to_npc`/`npc_to_robot`；选择按 actor-aware route cost 总和、
station ID 确定性排序。一次 lease 原子占用 station、分配的 role sites、
participants 和可选 object。session acquire 和 terminal release 的精确重放都不会
重复转换；release 必须提供 terminal receipt 和
`succeeded`/`failed`/`cancelled`/`timed_out` outcome。legacy v2 single-station adapter
仍被拒绝，不能作为 production evidence。

allocator 的 append-only journal 事件保存完整 lease/request/resource 证据、单调 sequence，
以及 release receipt/outcome。提供带内部 `RLock` 的内存 journal 和 durable JSONL
journal；JSONL 使用确定性序列化、严格字段解码和重复 JSON key 拒绝，append
在 flush/fsync 后才允许 allocator 提交内存转换。append 失败保留转换前状态。
reconstruction 重新校验 catalog role assignment、session signature、resource set 及事件顺序；
只有 active lease 恢复资源占用，已终止 session 保留幂等历史且不会重新 acquire。

该 allocator/journal 尚未接入 conversation/handover driver、agent runtime 或
simulation bridge，也尚未执行跨进程 file locking。因此这是 Phase 2A 的独立可审查
状态核心，不构成 runtime/production support；active catalog 和 20 个生产场景的
unsupported capability 声明未修改。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/agents/interaction_stations.py
  tests/test_interaction_station_allocator.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_interaction_station_allocator.py --tb=short`：`30 passed in 0.48s`。
- `git diff --check -- stretch_mujoco/agents/interaction_stations.py
  tests/test_interaction_station_allocator.py aaa_workspace/docs/current.md`：通过。

## Phase 2B NPC-NPC conversation station migration（未发布）

population schema v3 现可由 `OfficeAgentRuntime.from_json()` 真实解析，并从完整
`interaction_stations` catalog 构造 `InteractionStationAllocator`。embodied simulator assembly 将同一
allocator 注入 `MujocoNpcActionDriver`，并将该 driver 同时设为 runtime
`interaction_driver`。schema v2 仍仅保留可运行的 pair/site compatibility adapter，其
session、receipt 和 event 均标记
`compatibility_mode: population_v2_legacy_adapter` 与 `production_evidence: false`；该路径不会把
legacy single station 升格为多 station runtime support。schema v3 当前同样标记
`compatibility_mode: population_v3_candidate` 与 `production_evidence: false`，不构成已通过生产验证的证据。

Phase 2B 仅实现 NPC-NPC conversation。runtime 对 robot participant 稳定返回
`phase_2b_npc_conversation_only`，不会把 robot 交给 NPC command transport。conversation session
原子 acquire station/role-site/participant lease，`preferred_station_id` 可精确指定 station；
忙碌、无兼容 station 或 route unavailable 均保留 allocator 稳定错误。driver 不再为
v3 production conversation 降级到 pair-to-fixed-site map。

每个 phase 保留两个 participant 的确定顺序 terminal command receipt IDs。approach 后
双方按 leased role yaw 对齐，再各自执行 alignment gate；该 gate 同时验证自己仍处于
leased target site、authored distance min/max 及 mutual-facing tolerance。turn 不再只命令
speaker：speaker 必须获得 `talk_cycle` receipt，listener 同时必须获得 hold/gaze/
target-site receipt；任一失败都终止 session，不写 transcript、memory 或
`dialogue_turn_committed`。LLM future 路径未改为 physics-loop blocking wait。

`ConversationReceipt`、session memory 和 terminal/commit events 现显式携带 `station_id`、
`lease_id` 与完整 `physical_receipt_ids`。成功、失败、取消和超时都通过唯一
runtime cleanup 路径释放 participant 和 allocator resources，重放不重复发送 terminal event。
partial command submit 会取消已提交命令并回滚 lease，不留部分占用。

`RuntimeEvent.event_id` 在追加式 runtime audit history 中全局唯一；`tick()`/
`drain_events()` 只消费独立 pending 队列，不删除审计历史。动作物理成功时，
`action_succeeded` 保留自身 event ID，后续 `semantic_commit` 使用新 event ID，并以
`causation_id` 指向 `action_succeeded.event_id`；action memory 中的 `event_id` 仍精确对应
`action_succeeded`。

physical receipt 与 fail-closed cleanup evidence 严格分离。只有 poll 到当前命令自身的
`FAILED`/`CANCELLED`/`TIMED_OUT` terminal receipt，才会将该 command ID 写为
allocator `terminal_receipt_id`。显式 cancel 或 runtime deadline timeout 时，当前 transport API
不返回 cancellation terminal ack；因此使用 `release_unconfirmed(cleanup_evidence_id=...)`
释放资源，老的 approach/alignment receipts 只保留为前序链，不会被冒充为取消/
超时物理确认。

当前 runtime assembly 没有同步 live navmesh route estimator。因此 schema-v3 runtime 构造的
allocator route callback 明确返回 unavailable，production conversation 会 fail closed 为
`interaction_station_route_unavailable`。compiler navigation preflight 仍是 candidate build evidence，不会被伪装成
运行时动态可达性。注入实际 route callback 的 focused runtime 已验证双 station 并发，
但在 live estimator/cancellation ack 接入前不宣称 runtime 或 production supported。

验证：

- `.venv/bin/python -m py_compile stretch_mujoco/agents/actions.py
  stretch_mujoco/agents/conversation.py stretch_mujoco/agents/drivers.py
  stretch_mujoco/agents/interaction_stations.py stretch_mujoco/agents/runtime.py
  stretch_mujoco/agents/simulation_bridge.py stretch_mujoco/stretch_mujoco_simulator.py
  tests/test_conversation_station_runtime.py tests/test_office_agents.py`：通过。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_conversation_station_runtime.py tests/test_interaction_station_allocator.py
  tests/test_conversation.py tests/test_npc_stretch_e2e_acceptance.py
  tests/test_semantic_action_bindings.py tests/test_interactions.py`：`86 passed in 55.04s`。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_office_agents.py::test_action_success_causes_distinct_semantic_commit_event
  tests/test_conversation_station_runtime.py::test_success_uses_one_cleanup_path_and_does_not_double_emit`：
  `2 passed in 0.50s`。
- `git diff --check`：通过。
- active catalog SHA-256 仍为
  `bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365`；20 个
  office/home production configs 仍有 40 条 `unsupported`。

## Phase 2C NPC-NPC handover station migration（未发布）

`MujocoNpcActionDriver` 的 population-v3 NPC-NPC handover 现从同一
`InteractionStationAllocator` 原子 acquire station、giver/receiver role sites、双方 participant
和 object claim。候选 station 仍依 route callback 总成本和 station ID 确定选择，
`preferred_station_id` 从 `ActionCommand.parameters` 贯穿到 allocator。忙碌 station/site/
participant/object 和 route unavailable 继续使用 allocator 的稳定 fail-closed 错误，
任一 acquire 失败都不保留部分资源。

lease assignment 是 giver/receiver site 和 authored yaw 的唯一 v3 来源；station 的
`distance_m`、`yaw_tolerance_rad` 用于双方 readiness marker gate，DETACH payload
显式使用 authored `transfer_site`。成功链为双方 approach receipts -> 双方 align
receipts -> giver/receiver `handover_ready` marker receipts -> giver DETACH receipt -> receiver
ATTACH receipt -> `pull_npc_states()` 唯一 observed receiver owner -> allocator terminal release ->
runtime semantic commit。命令提交或 duration 不会被当作 marker、attachment observation 或
terminal receipt。

rendezvous/aligned/ready 的双命令提交共用 handover 自身的 partial-submit
rollback：第二条 submit 抛错会取消已提交命令，以 unconfirmed cleanup evidence
释放 lease/object claim，且 cleanup ID 不进入 physical receipt chain。release/receive/
rollback submit exception 也同样 fail closed。如 DETACH 已成功而 receiver ATTACH 未成功，
receive failure、cancel 或 timeout 都先尝试 giver reattach reconciliation；只有 rollback
terminal receipt 成功才记录恢复。rollback submit/receipt 失败会记录
`handover_reconciliation_required` cleanup evidence，不宣称 owner 已恢复，也不提交
receiver 语义 owner。

handover `DriverResult`、`ActionExecution`、action memory 和 runtime events 保留 station/lease、
完整有序 physical receipt IDs、cleanup evidence、compatibility mode 和
`production_evidence: false`。`action_succeeded` 以最后 attachment terminal receipt 为
causation，后续唯一 ID 的 `semantic_commit` 再指向 `action_succeeded.event_id`。
重放同一 terminal execution ID 返回原结果，不二次 transfer 或 reacquire。

population v2 旧 pair-to-fixed-role map 仍可运行，但 receipt/event 明确标记
`compatibility_mode: population_v2_legacy_adapter` 和 `production_evidence: false`。legacy
conversation pair 已恢复 `(speaker.site, listener.site)` 的顺序映射：first participant ->
speaker，second participant -> listener。MuJoCo E2E 临时 fixture 按实际可达路径把 Priya
设为首轮 speaker、Jordan 设为 listener，不再以反转角色映射规避路径。

Phase 2C 仅实现 NPC-NPC。`npc_to_robot`/`robot_to_npc` 和 robot participant 在命令
进入 NPC transport 前稳定拒绝为 `phase_2c_npc_handover_only`。生产 runtime
assembly 仍没有 live route estimator，v3 allocator 因此在 acquire 阶段返回
`interaction_station_route_unavailable`。没有修改 active catalog 或 capability status，不宣称
runtime/production supported。

验证：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_handover_station_runtime.py --tb=short`：`20 passed in 0.52s`。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_handover_station_runtime.py tests/test_action_driver.py
  tests/test_npc_runtime_safety.py tests/test_interaction_station_allocator.py
  -k 'not test_production_driver_requires_scene_binding_for_registered_gesture_point'
  --tb=short`：`87 passed, 1 deselected in 0.58s`。deselected 项仍是 Phase 0 已记录的
  `ActionType.IDLE` expectation mismatch，本阶段未修复。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_conversation_station_runtime.py tests/test_interaction_station_allocator.py
  tests/test_conversation.py tests/test_npc_stretch_e2e_acceptance.py
  tests/test_semantic_action_bindings.py tests/test_interactions.py`：`86 passed in 50.11s`。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_npc_stretch_e2e_acceptance.py --tb=short`：`1 passed in 49.13s`。
- `.venv/bin/python -m py_compile stretch_mujoco/agents/actions.py
  stretch_mujoco/agents/drivers.py stretch_mujoco/agents/runtime.py
  tests/test_handover_station_runtime.py tests/test_npc_stretch_e2e_acceptance.py`：通过。
- `git diff --check`：通过。
- active catalog SHA-256 仍为
  `bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365`；20 个
  office/home production configs 仍有 40 条 `unsupported`。

## Phase 4 minimal seat-slot and desk-work runtime slice (2026-09-17)

This worktree now has a private population-v3 seat-slot execution path. The
immutable interaction catalog preserves seat owner/type/index/clearance and
workstation/computer/seat bindings. `SeatSlotAllocator` reserves and occupies
the slot ID itself under an `RLock`: separate slots on one sofa are independent,
while a second session cannot reserve an already reserved or occupied slot.

For a v3 slot, `MujocoNpcActionDriver` executes ingress movement, authored-yaw
alignment, the `seated` animation marker, and an injected seat contact/collision
verification callback. Only the complete physical receipt chain marks the slot
occupied. `STAND_UP` retains occupancy until its marker, optional ingress exit,
and standing verification receipt all succeed. `MOVE_TO` remains fail-closed
while the driver records the NPC as seated. A missing verifier fails closed;
the callback is an integration boundary and this slice does not manufacture a
MuJoCo contact receipt. Legacy Chair execution remains a compatibility adapter.

`USE_COMPUTER` resolves `Computer --ON--> Workstation` and then exactly one
`SeatSlot --NEAR--> Workstation`. The existing desk-work session lowers it to
`SIT -> WORK(work_cycle marker) -> STAND_UP -> IDLE`; command parameters and
runtime event/memory evidence carry computer, workstation, slot, session, and
the accumulated physical receipt IDs. A work failure leaves the already queued
safe stand/idle path available; a stand failure retains the seated slot and
semantic occupancy rather than claiming release.

Focused evidence:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_seat_work_runtime.py`: 5 passed.
- Phase 4 plus action-driver compatibility: 39 passed and the unchanged
  pre-existing `test_production_driver_requires_scene_binding_for_registered_gesture_point`
  failure caused by the already-present extra `ActionType.IDLE`.
- Two legacy desk-work/conversation interruption regressions: 7 passed after
  retaining the original Chair `MOVE -> SIT` adapter.
- Final Phase 2 + Phase 4 regression command covering the new focused file,
  conversation station runtime, allocator, conversation, MuJoCo E2E, semantic
  bindings, interactions, and handover station runtime: 111 passed in 51.48s.
- `py_compile` for all five touched source modules and the focused test passed;
  `git diff --check` passed. The active catalog SHA-256 remains
  `bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365`,
  and the office/home production configs still contain 40 `unsupported`
  declarations.

Limitations remain explicit: this is not production evidence, generated scene
projections and active capability declarations are unchanged, REST is not
migrated in this minimal slice, and no real Stretch robot executor or Phase 5/6
authoring was added. Population-v3 live use still requires the simulator owner
to provide a real `verify_seat_contact` callback.

## Phase 5 candidate authoring for twenty scenes (2026-09-17)

The twenty office/home source scenes now have deterministic
`interaction_site_plan/v1` candidates and `scene_npc_config/v2` configs. The
authoring path is owned by `tools/author_scene_interaction_plans.py` and
`tools/propose_interaction_sites.py`: each scene loads its own manifest and
MJCF, builds its own NPC navmesh, selects six distributed same-region pairs,
and emits three NPC-NPC conversation plus three NPC-NPC handover stations. No
office_02 coordinates are copied to other scenes.

Seat authoring closes every manifest seating entity with either a local
navmesh-backed slot or a named exemption. Across the twenty scenes the current
candidate inventory is 136 seat slots and 13 exemptions. Every scene has one
physically bound workstation/computer/seat binding. Homes outside the three
native workstation scenes (home 03/05/06) receive an idempotent,
redistributable MuJoCo primitive desk/computer/chair/cup kit in the source MJCF
and manifest, with `primitive_workstation_kit/v1` provenance and CC0 metadata.
Handover objects are manifest-backed, dynamic, graspable free-joint bodies.

`tools/generate_scene_npc_resources.py` now emits all twenty configs as v2,
references the authored plans, and declares only the scoped NPC capabilities:
conversation, handover, sit, and work. The previous 40 conversation/
robot-handover `unsupported` placeholders are absent. Compiling these configs
produces population schema v3 projections; generated projections were not
manually edited and the active catalog was not published or changed.

`tools/validate_interaction_site_plan.py` and the interaction-site audit mode
report schema, source binding, plan hash, counts, and NPC navmesh evidence.
Robot-NPC conversation, robot handover, robot footprint preflight, and a real
Stretch executor are explicitly `deferred_out_of_scope`; they are not NPC
completion or publication gates. Static collision/contact and per-seat visual
or animation acceptance remain Phase 6 work, so every record remains
candidate-only with `production_evidence: false`.

Observed inventory and checks:

- 20 plans and 20 v2 configs; 60 conversation stations, 60 NPC-NPC handover
  stations, 136 seat slots, 13 named exemptions, and 20 workstation bindings.
- `unsupported` occurrences in the twenty configs: 0.
- `tests/test_twenty_scene_interaction_plans.py`: 23 passed independently.
- Applicable v2 schema contract plus the batch suite: 47 passed in 71.16s.
- `py_compile` for the author/proposal/validator/generator/audit tools, plan
  loader, and batch test passed; `git diff --check` passed.
- Active catalog SHA-256 remains
  `bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365`.

The earlier office_02 pilot tests that hard-code one conversation, one
handover, robot modes, old resource IDs, and conversion from a v1 config are
superseded by this 3+3 NPC-only authored catalog. They require fixture migration
before being used as Phase 5 evidence; runtime physical acceptance is not
claimed from the static batch suite.

## Production V2 fitted no-hair appearance bundle (2026-09-17)

The V2 fitted no-hair proposal is preserved for future review but is **not**
the production default: the ten-NPC production catalog uses the original
`smplx_office_neutral_v1` bundle and its original appearance identities. The
versioned V2 recipe is
`models/appearance_recipes/office_personas_v2_fitted_no_hair.runtime.json`, and
`tools/build_npc_v2_fitted_appearance.py` reproducibly projects all 215 unique
source animation frames into a stable fused topology. Each frame retains the
SMPL-X body and appends UV-preserving surface shells for the top, trousers and
shoe uppers at 10 mm, 9 mm and 12 mm offsets. It also appends 12 mm thick,
foot-outline soles with a narrowed arch and heel. Keeping these parts in each
body OBJ avoids MuJoCo's independent-mesh recentering problem.

The V2 catalog adds `hair_none_v1`, a transparent layer used only by its ten
versioned identities. Its glasses remain the original `glasses_thin_round_v4`
`accessory_2d` texture layer for Alex, Priya, Daniel and Lena; no geometric
glasses are generated or loaded. The original `smplx_office_neutral_v1` bundle
and original hair/appearance identities are the active production selection.
All office/home scene configs continue to reference
`office_population.production.example.json`, so future scene compilation uses
that restored default rather than embedding the V2 proposal.

Generated local runtime assets live under
`models/assets/humanoid/generated/animations/v2_fitted_no_hair_v1/` (215 OBJ
frames, about 371 MiB) with a hash receipt. They remain ignored local projections
and can be rebuilt from the tracked recipe/tool; licensed source assets are not
committed. The manifest validates 10 NPCs and 2 bundles. Focused tests
`tests/test_npc_production_appearance_roster.py tests/test_npc_assets.py` pass
(`14 passed`). A composed native-office MJCF compiles to 70 bodies, 2,626 geoms,
2,244 meshes, 27 textures and 61 materials. A 40-frame Alex front/rear/left/
right/top walk-and-sit acceptance render passes with scene-native lighting and
no color, transparency or occlusion failures.


## 2026-09-17 Live MuJoCo representative seat acceptance (not production)

The candidate-static plans were exercised against all twenty compiled MJCF scenes using
the in-process NpcSystem transport, real MuJoCo model/data, mj_step,
controller-issued move/align/sit/stand commands, animation completion markers, live
root-to-site measurements, and MuJoCo contact scans. The reports live under
aaa_workspace/archive/20260917_repository_cleanup/aaa_workspace/interaction_acceptance_physical/.

This is a representative per-scene seat check, not an exhaustive 136-slot campaign:
one deterministic authored seat slot was selected per scene. All 20 reports are
physical_mujoco_candidate, retain production_evidence: false, and the aggregate
result is passed: false. No active catalog was published.

Observed blockers are physical and fail closed: 16 scene checks ended
route_static_collision, 4 ended route_unavailable; office_01 also observed an NPC
torso penetration against 075_bread_022_collision. The current mocap mesh NPC asset
has no force-bearing chair proxy, so seat evidence is explicitly limited to a live
root-to-sit-site tolerance plus a clean MuJoCo unexpected-contact scan. It must not
be relabeled as force contact.

Verification:

- PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  tests/test_candidate_interaction_acceptance.py tests/test_seat_work_runtime.py --tb=short:
  7 passed in 2.34s.
- PYTHONPATH=. .venv/bin/python tools/run_scene_interaction_acceptance.py --all
  --live-mujoco --output-root aaa_workspace/interaction_acceptance_physical:
  completed 20 reports, aggregate passed: false.
- .venv/bin/python -m py_compile tools/run_scene_interaction_acceptance.py and
  git diff --check: passed.

The next physical-correction slice is to resolve each reported ingress/seat route,
move or exclude overlapping dynamic props at spawn, add a force-bearing NPC seat
proxy if contact-level acceptance is required, then rerun all seat slots rather than
only the representative slot.

## LLM local-decision profile context (2026-09-17)

`OfficeAgentRuntime.queue_llm_event()` now adds the target NPC's authoritative
profile to every queued LLM event. In addition to the existing day-start schedule
context, `new_task`, `dialogue`, `repeated_failure`, `unexpected_change`, and
`reinterpret_plan` requests therefore receive `profile.role`,
`profile.department`, `profile.personality`, and `profile.preferences` alongside
their event-specific context. Values are copied from the runtime-owned
`EmployeeProfile`; caller-supplied values with the same keys cannot impersonate
another personality or preference set.

This changes only LLM decision context. It does not allow LLM responses to mutate
the profile, does not change deterministic utility scoring or needs maintenance,
and does not move provider calls into the physics loop. Focused verification:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
tests/test_low_compute_behavior.py tests/test_llm_provider.py --tb=short`: `23 passed`.

## Tracked showcase dependency closure (2026-09-17)

`tools/render_active_scene_showcase.py` no longer imports helper functions from
the ignored local `aaa_workspace/demo_new/` tree. Its portable-MJCF and route-audit
helpers are now owned by the tracked tool, so a fresh Git checkout has the same
source dependency closure as this primary worktree. The remaining ignored Python
and shell files under `aaa_workspace/{demo_new,experiments,raw_resources}` are
local demo, experiment, or intake tooling and are not imported by tracked runtime
or tools. Verification uses `py_compile`, a tracked-files-only archive import,
`git diff --check`, and the focused LLM tests recorded above.

## Models resource cleanup (2026-09-17)

The model tree now keeps `stretch_mujoco/models/assets/` as the canonical payload
root. About 211 MiB of superseded or generated local material was moved, not
deleted, to the ignored
`aaa_workspace/archive/20260917_models_cleanup/` directory. The archive contains
the unreferenced tracked `temp_old_sg3_assets` meshes, two recursive symlinks,
generated hidden acceptance MJCFs, the unused `models/humanoid/` placeholders,
legacy root-level resource directory copies, and unreferenced placeholder PNG
copies. `models/assets/wood.png` remains because `graspgen` references it.

The compatibility projections in `generated_home_npc/` and
`generated_office_npc/`, the single active content-addressed build, HSSD caches,
and manifest-backed humanoid assets remain in place. No active configuration or
catalog path was changed.

After phase 1, the asset, appearance, generated-home, and generated-office suites
passed (`35 passed in 1013.43s`). After phase 2, MuJoCo loaded `scene.xml`,
`office_scene.xml`, and all twenty active office/home MJCFs. The separate legacy
`office_scene2_multi_npc.xml` wrapper still fails on its pre-existing missing
`assets/office_scenes/stretch/base_link_0.obj` path; no archived path matches or
previously supplied that location.
