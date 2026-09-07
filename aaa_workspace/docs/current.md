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

`hair_layers.py` 复用已烘焙 idle OBJ 的顶点与 UV，只栅格化高于设定 head height 的头部面，生成短发发帽层。它当前提供 `employee_short_black_v1`、`employee_short_brown_v1`、`employee_short_dark_blonde_v1` 和 `employee_short_auburn_v1` 四种可登记 appearance；示例 spec 位于 `appearance_sources/short_hair_v1/hair.json`。这四个外观只改变同一短发覆盖区域的颜色，不能增加发丝、长发轮廓或物理摆动。生成与登记命令为：

```bash
uv run make_npc_short_hair_layers --spec .../short_hair_v1/hair.json --output-dir .../short_hair_v1/layers
uv run bake_npc_appearance --recipe .../employee_short_black_v1.recipe.json \
  --output-dir .../appearances/employee_short_black_v1 \
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

当前 catalog 支持任意顺序的同 UV 2D layers（例如 skin、freckles、face_detail、hair、top、bottom、shoes）；它不承诺防止两个绘制者选择语义冲突的 layer。该类组合策略由内容制作约束管理。2D 眼镜只是脸部贴花；帽子、背包等独立 OBJ 则走 manifest-backed `accessories` 加逐帧 anchor 的路径。

### 3.0.2 外部 OBJ 配饰预处理（preview）

文件：`tools/prepare_obj_accessory.py`

`prepare_obj_accessory.py` 是一个可复现的本地预处理工具：它从来源归档解出 OBJ，规范化为 MuJoCo 的 metre/Z-up 网格，按每段动画的头顶顶点生成 anchor，并输出 mesh、anchor 与 SHA-256 receipt。当前来源 cap 的输入约定是嵌套归档的 `source/cap.zip` 与其中的 `cap.obj`；工具采用 centimetre/Y-up 到 metre/Z-up 的转换，输出网格将水平包围盒居中且最低点置为局部 `z=0`，逐帧 anchor 才是实际头顶位置。当前 cap preview 默认 `--mesh-scale 1.3 --head-clearance-m -0.026 --back-offset-m 0.098 --back-tilt-degrees 13`：网格等比放大 1.3 倍，最低点相对头顶下移 2.6 cm，沿局部 `+Y`（人物面向为 `-Y`）后移 9.8 cm，并绕局部 X 轴向后倾 13°；这些值会写入 receipt，可按模型实际形状调整。原始归档及其派生网格均属本地 preview 资产，不提交、也不会自动进入生产 manifest。

```bash
uv run python tools/prepare_obj_accessory.py \
  --source-archive aaa_workspace/raw_resources/objs/cap.zip \
  --manifest stretch_mujoco/models/assets/humanoid/generated/animations/manifest.json \
  --output-dir stretch_mujoco/models/assets/humanoid/generated/animations/accessories \
  --accessory-id cap_source_v1
```

要启用该 preview，必须在副本 manifest 的 `accessories.cap_source_v1` 中显式写入 mesh、anchors 和对应 SHA-256，然后让 NPC `appearance_config.accessories` 与 `embodiment.accessories` 同时引用它；缺少 hash、文件或某 clip/frame anchor 均会被 preflight 或 scene build 拒绝。验收使用 `tools/render_npc_personas.py` 的每 NPC `front`/`side`/`sit` plan；其切帧函数只显示同一 clip/frame 的 body 与 accessory geoms，避免相邻透明帧叠加或闪烁。视频是观察证据，此外必须用 manifest 校验确认无缺失资源，并检查三视图中的头皮/脸/耳部遮挡和颜色稳定性。

当独立 OBJ 在不同动画 clip 中出现相对漂移时，使用 `tools/fuse_obj_accessory.py` 生成每帧的 `body + accessory` OBJ，而不是继续调 anchor。该工具以 body frame、规范化 accessory mesh 和逐帧 anchor 为输入，以 `idle/0` 的头部顶点为参考，对每个同拓扑 body frame 做 Kabsch 刚体头部对齐（平移和旋转），然后写入融合 frame。输出是只含融合 frame 路径的 preview manifest；运行时必须从人口配置中移除该 accessory，避免再创建第二个独立 geom。融合网格是该 preview 的唯一运行时投影，因此 body 和帽子同受 MuJoCo 的同一 mesh 中心化与 alpha 切帧，并且随头部的坐姿、站姿和行走变换共同运动。原始 body UV 保持不变；当前 cap face 使用一个固定 atlas UV，专用帽子 atlas 仍是后续美术资产工作。

### 3.0.3 融合 OBJ 配饰 recipe（当前事实源）

文件：`stretch_mujoco/npc/appearance_pipeline/accessory_recipe.py`、`tools/build_npc_fused_accessory.py`

融合 OBJ 的可编辑事实源是 schema-v1 recipe，而不是 `*.receipt.json`。recipe 固定 `accessory_id`、`attachment_mode: head_follow_fused`、`mesh_scale`、`head_clearance_m`、`back_offset_m` 和 `back_tilt_degrees`；仓库只提供无私有来源路径的 `models/accessories/cap_source_v1.recipe.example.json` 模板。调用方显式传入本地来源归档、源 manifest 与源 population，builder 才生成 accessory、逐帧 head-follow fused OBJ、receipt、fused manifest 和 runtime population projection。

```bash
uv run python tools/build_npc_fused_accessory.py \
  --recipe stretch_mujoco/models/accessories/cap_source_v1.recipe.local.json \
  --source-archive aaa_workspace/raw_resources/objs/cap.zip \
  --source-manifest stretch_mujoco/models/assets/humanoid/generated/animations/manifest.json \
  --source-population stretch_mujoco/models/office_population.production.example.json \
  --npc-id employee_01 \
  --output-dir stretch_mujoco/models/assets/humanoid/generated/animations/fused/cap_source_v1 \
  --output-manifest stretch_mujoco/models/assets/humanoid/generated/animations/manifest.cap_source_v1.preview.json \
  --output-population stretch_mujoco/models/office_population.cap_source_v1.preview.json
```

Builder 将 recipe SHA、receipt SHA 和融合模式写入输出 manifest，并从 target NPC 的 `embodiment.accessories`（及出现时的 `appearance_config.accessories`）移除该 `accessory_id`。随后严格 manifest/population preflight 必须通过；因此运行时只能加载融合帧，既不能静默复用旧 anchor，也不能再次创建独立帽子 geom。receipt 的 pose 字段由 builder 在写出时与 recipe 精确比对；手工修改 receipt 不会改变 runtime projection，下一次 build 会重新覆盖它。

运行时场景组合必须使用 `tools/build_npc_scene.py --accessory-runtime-config`，而不是把旧的 fused manifest 直接传给 `build_npc_scene()`。runtime config 将可编辑 recipe、原始归档、基线 manifest/population 和派生输出位置集中在同一份本地 JSON；每次调用都会无条件重新生成 OBJ、anchors、逐帧融合 OBJ、receipt、manifest 和 runtime population，随后才组合 MuJoCo 场景。它没有 receipt/mtime/cache 快路径：receipt 仅为本轮投影的审计证据，不能成为运行时输入。配置路径相对 runtime config 文件解析；示例 `models/accessories/cap_source_v1.runtime.example.json` 不包含私有资源且仅作本地配置模板。

```bash
uv run python tools/build_npc_scene.py \
  --accessory-runtime-config stretch_mujoco/models/accessories/cap_source_v1.runtime.local.json \
  --output stretch_mujoco/models/.office_npc_runtime.xml \
  --include-base-scene
```

该 CLI 是 recipe-bound OBJ 配饰的唯一受支持运行入口。`--population` 保留给没有 recipe-bound OBJ 配饰的既有兼容场景；它不会猜测来源归档或从 receipt 重建资产。任何 recipe、来源或严格预检错误都会在 MuJoCo 加载之前终止，避免静默复用旧模型。

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

验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl uv run pytest -q tests/test_recording.py` 通过（5 passed）；直接运行 pytest 会被环境自动发现的 ROS `launch_testing` 插件阻塞，原因是该环境缺少 `lark`，并非此渲染器测试失败。

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

`face_details.py` 现在按 face mask 的每个实质性、不连通 UV 岛分别绘制细节。先前将多个岛合并为一个边界框，会把眼镜等图案画进透明 UV 接缝；新测试确保双岛 mask 的两侧都获得贴图内容。局部离屏审查使用新增的 `tools/render_npc_personas.py`：它先通过 `NpcPopulation`、catalog 与 asset manifest 验证，再在临时的 standalone MJCF 中加入仅供审查的灯光和地面，输出 H.264 MP4；它不修改生产 scene。

复现命令：

```bash
MUJOCO_GL=egl .venv/bin/python tools/render_npc_personas.py \
  --population stretch_mujoco/models/office_population.production.example.json \
  --output outputs/npc_persona_accessories.mp4
```

首次审查发现旧 face mask 把局部正脸方向反选为 `Y >= 0`，配饰落在头后；旧短发只按高度选择，覆盖了鼻子和耳朵。该输出不再作为推荐身份：新增 face mask v2 使用 `Y <= -0.02` 且 outward normal 的 Y 分量不大于 `-0.35` 选择正脸；短发 v3 额外要求 upward normal，且将最低高度提高到 `1.67m`。重新生成、烘焙并离屏复核后，Alex 的眼镜、眉毛和胡须在正脸，Morgan 的发帽止于额头上方。雀斑是低不透明度贴图，当前审查灯光下较淡。

第二次审查又收窄了 facial decal：仅向不小于主 face island 一半面积的岛绘制，避免把胡须和眼镜重复到 neck/seam island；胡须从口部下移为下巴短胡须，镜圈缩小为细框，并加入贴图式鼻梁和镜腿线。正脸复核中嘴唇不再被胡须覆盖。镜腿在正视时天然不明显，且仍仅为 UV 像素，不会形成侧面可见的实体镜架或碰撞形状。需要可见轮廓、物理碰撞或更精细发型时，仍应引入独立 geometry slot，而不是把 UV decal 宣称为 3D 配饰。

在用户明确要求清理后，所有经渲染证明无效或只绑定旧人物名称的 appearance、face detail、short hair 和 semantic mask v1–v3 资源已从本机的 ignored `generated/` 目录移入系统回收站。catalog 与 manifest 同步收缩：保留 `employee_default_v1`、`office_warm_base_v1`、`office_warm_glasses_chin_beard_v1`、`office_short_brown_freckles_v1` 四个 appearance，以及 `semantic_masks_v2`、`short_hair_v3`、`face_details_v4` 的来源。删除的是可重新生成的本地派生文件，不影响版本库中受跟踪的代码或基础资产。

## 14. 独立 OBJ 配饰（第七轮，进行中）

`NpcEmbodiment.accessories` 现在接受通用 accessory ID。asset manifest 的 bundle 可声明每个 accessory 的 OBJ 与逐 clip/逐 frame anchor JSON；二者都由 SHA-256 校验。scene builder 为人体每个 clip frame 同时生成同名帧的 accessory geom，`MeshSequenceBackend` 因而会在切换 idle、walk、sit 帧时同步切换人体和配饰，不会把眼镜固定在 mocap 根节点。

首个本地 manifest-backed demo 现为 `cap_simple_v1`。先前的 `glasses_thin_round_v1` 已被移入系统回收站。`tools/generate_cap_accessory.py` 从已验证的每帧人体 OBJ 推导头顶 anchor，并生成 OBJ 与 anchor JSON；`employee_01` 通过 `accessories: ["cap_simple_v1"]` 选择它。以下命令渲染 idle、walk、sit 三段近景验收：

```bash
MUJOCO_GL=egl .venv/bin/python tools/render_npc_personas.py \
  --population stretch_mujoco/models/office_population.production.example.json \
  --output outputs/npc_cap_obj_demo.mp4
```

审查首版后，矩形帽冠和帽檐已替换为更简单的 `12` 边圆形帽檐（半径 `0.095m`）与低多边形圆顶帽冠（半径 `0.078m`、高度 `0.052m`）。逐帧 anchor 维持头顶最大高度下方 `0.01m`：MuJoCo 会重心化 OBJ 的局部 bounds，先前把 anchor 上抬 `0.025m` 的做法会让帽子明显悬空；这一微小 inset 则让帽檐在 idle、walk、sit 可见帧均贴合头顶，且无可见头皮穿透。每次重新生成后必须同步更新 manifest 中 OBJ 和 anchor 的 SHA-256，否则 asset validation 会拒绝场景构建。

当前 OBJ 是可验证的低多边形 demo，只有渲染 geom、无碰撞和物理交互；它不等同已完成艺术资产。后续正式配饰应替换 OBJ、重新生成每帧 anchor 并更新 manifest hash。

## 15. 显式外观配置与验收闭环（第八轮，进行中）

`NpcAppearance` 是 JSON 中绑定到 `NpcDefinition.agent_id`（未填写时等于 `npc_id`）的显式配置，固定选择 `skin`、`hair`、`top`、`bottom`、`shoes`、`accessories` 和 `scale`。它作为 `embodiment.appearance_config` 与既有 `appearance`/`visual_identity` 并存：旧 population 可继续加载；声明 catalog 的新配置会验证每个语义 slot 是该 identity 已选择、且 category 相符的 layer。运行时仍将这些 layer 烘焙为单一 body atlas，因此这不是 split-mesh material 已完成的声明。

烘焙 recipe 现接受 32-bit `seed`（默认 `0`），catalog identity 未声明 seed 时从 catalog 字节和 identity ID 稳定导出。每次烘焙在原有 hash sidecar 中写入 seed、recipe hash、thumbnail 名称与 thumbnail hash，并生成不大于 `128px` 的 PNG 缩略图；这份 `*.appearance.json` 是统一的可再现 metadata receipt。PNG/OBJ/缩略图是本地派生物，仍不提交。

`render_npc_personas.py` 现为 population 中每个 NPC 排入 front/idle、side/idle、sit/sit 三个验收镜头。frame visibility 以解析后的 frame geom 名称为事实源，每次只显示同一 clip/frame 的 body 和 accessory geom；alpha 回归测试验证不会同时暴露相邻帧，避免透明帧切换闪烁。实际离屏渲染仍要求本地已验证的 SMPL-X 生成资产。

本轮验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_npc_schema.py tests/test_npc_appearance_catalog.py tests/test_npc_persona_review.py tests/test_npc_face_details.py tests/test_npc_hair_layers.py tests/test_npc_semantic_masks.py` 得到 `18 passed`；对 schema、bake、catalog 和 renderer 的 mypy 检查通过。以本地 production population 运行 `MUJOCO_GL=egl .venv/bin/python tools/render_npc_personas.py ...` 成功生成 120 帧 acceptance MP4，并抽查 Alex 的侧面和两个 NPC 的坐姿段。完整 tests/ 未在本轮运行。
