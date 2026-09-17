# NPC 系统使用指南

本文件是办公室与家庭场景 NPC 系统的使用入口。当前事实源是严格的
`scene_npc_config/v1` 配置和 active 编译目录；它取代了早期按办公室/家庭分别
预处理的 NPC 派生文件。实现与验证记录见
[`aaa_workspace/docs/current.md`](../aaa_workspace/docs/current.md) 的“办公室/家庭统一语义场景编译”章节。

## 1. 先确认什么是“有效配置”

运行时应从以下链路加载，不应直接挑选旧的 `*_npc.xml` 或手改生成 JSON：

```text
scene_npc_configs/catalog.json
        │ 列出 20 个受控场景源配置
        ▼
scene_npc_config/v1（每场景的语义、人口、业务路线）
        │ + source MJCF/manifest + semantic policy + NPC catalog
        ▼
generated_scene_npc/active/active_catalog.json
        │ 指向一个不可变的内容寻址 build
        ▼
<build>/<scene_id>/{*_npc.xml, population, semantic.v2,
                   semantic.v1, semantic_coverage, trajectory_profile, receipt}
```

截至本文档更新时，active build 是
`6ecc0ea9c8d86957031ed5ebac761c8d52a31fac9d039d54b48462a6e30896ff`，入口为：

```text
stretch_mujoco/models/generated_scene_npc/active/active_catalog.json
```

它包含全部 20 个场景：10 个办公室和 10 个家庭。全量产物中共有 1,310 个已注册
语义实体、1,350 个 required navigation point、62 个 NPC 实例和 4,100 条路线；
`unresolved`、`unbound`、`unreachable`、显式豁免均为 0，所有场景均强连通。

`active_catalog.json` 只存 scene ID 和 build 根目录；加载某一场景时，先读取其
`build_root`，再拼接 `<build_root>/<scene_id>/`。例如 office_02 的有效 population 是：

```text
stretch_mujoco/models/generated_scene_npc/active/builds/
  6ecc0ea9c8d86957031ed5ebac761c8d52a31fac9d039d54b48462a6e30896ff/
  office_02_cross_axis/office_02_cross_axis.population.json
```

生成目录是可验证投影，不是编辑入口。修改必须从第 4 节的源配置开始，然后重新发布
active build。

## 2. 当前场景、人口与业务路线

所有有效源配置由
`stretch_mujoco/models/scene_npc_configs/catalog.json` 列出。办公室配置在
`office/`，家庭配置在 `home/`。

| 场景 | 人数 | 当前 NPC |
| --- | ---: | --- |
| `office_01_linear_bench` | 3 | Alex、Morgan、Jordan |
| `office_02_cross_axis` | 4 | Alex、Priya、Morgan、Jordan |
| `office_03_long_gallery` | 2 | Alex、Morgan |
| `office_04_central_meeting` | 4 | Alex、Priya、Morgan、Jordan |
| `office_05_team_clusters` | 3 | Alex、Priya、Jordan |
| `office_06_diagonal_flow` | 3 | Alex、Morgan、Jordan |
| `office_07_u_bench` | 3 | Alex、Priya、Morgan |
| `office_08_dual_island` | 4 | Alex、Priya、Morgan、Jordan |
| `office_09_staggered_rows` | 2 | Alex、Jordan |
| `office_10_social_core` | 4 | Alex、Priya、Morgan、Jordan |
| `home_01_102344115` 至 `home_10_104862513_172226580` | 各 3 | Alex、Jordan、Morgan |

NPC 的 canonical ID 分别为 `npc_alex_chen`、`npc_jordan_patell`、
`npc_morgan_lee` 与 `npc_priya_narayanan`。当前 62 个实例的分布为 Alex 20、Jordan
18、Morgan 18、Priya 6；其余 6 个 catalog 人设可供后续场景选用，但尚未出现在 active
scene roster 中。

每个场景另外都有两条显式业务路线：办公室为 `work_to_meeting` 和
`meeting_to_snack`；家庭为 `living_to_kitchen` 和 `bedroom_to_living`。编译器会再从
每个 NPC spawn 自动生成到所有 required navigation point 的 coverage route。覆盖路线不应
手写或删改，它们是“每位 NPC 都能到达每个有意义区域”的可验证保证。

## 3. 资源地图与职责

| 资源 | 位置 | 用途与编辑规则 |
| --- | --- | --- |
| 有效场景清单 | `models/scene_npc_configs/catalog.json` | 20 个源配置的唯一清单；新增场景后才加入。 |
| 单场景配置 | `models/scene_npc_configs/office/*.json`、`home/*.json` | 唯一的语义点、人口、业务路线、导航和失败策略编辑入口。 |
| 办公室人口方案 | `models/scene_npc_configs/office_population_plans.json` | 各办公室 2–4 人 roster 的共享输入。 |
| 家庭 slot 初始资源 | `models/assets/home_scenes/npc_slot_overrides.json` | 家庭历史 slot 的可审计输入；生成后坐标已写回每户配置。 |
| 场景基础资产 | `models/assets/office_scenes/`、`models/assets/home_scenes/` | 原始 MJCF 和 manifest；配置引用它们，编译器不修改源 XML。 |
| 分类 policy | `models/semantic_policies/office.json`、`home.json` | 将 manifest/XML 实体映射为 semantic class、affordance 和点位 bundle。 |
| 家庭实例例外 | `models/semantic_policies/home_scene_overrides.json` | reviewed 的每户 entity override 入口；当前为空。不能用无理由豁免绕过 coverage。 |
| NPC catalog | `models/office_population.production.example.json` | active 配置目前引用的 schema-v2 roster/embodiment 输入；含 10 位经 catalog 描述的人设。 |
| 资产与外观 | `models/assets/humanoid/`、`models/appearance_recipes/`、`models/accessories/` | manifest、appearance catalog、动画、配饰和受限资产的归属位置。配置只能引用 manifest 中已验证的 bundle/appearance。 |
| 家庭 slot/room 溯源 | `models/generated_scene_npc/provenance/home_slot_migration.json`、`home_room_overlays.json` | 记录 legacy slot 到可达格点的投影，以及由房间 hint 得到的 room seed；只读审计资源。 |
| active 产物 | `models/generated_scene_npc/active/` | 正式加载入口与不可变 build；只能由 audit 发布。 |
| fixture | `models/scene_npc_configs/fixtures/minimal_scene.*` | schema/compiler 测试样例，不能登记到 active catalog。 |

仍保留但不是办公室/家庭的当前事实源的资源：

- `models/generated_office_npc/`、`models/generated_home_npc/`：旧的预处理投影，供兼容测试、
  demo 或历史流程使用；不得作为新增语义/路线/人口的编辑入口。
- `models/office_population.json`：preview 示例；
  `office_population.production.example.json` 是可复用 roster/资产 catalog 输入，不等同于
  某一个 active 办公室场景。
- `npc/trajectory_profiles/office_v1.json` 和 MJCF `npc_trajectory_profile` custom text：
  legacy fallback。新统一场景使用 active build 中同场景生成的
  `*.trajectory_profile.json`。

## 4. `scene_npc_config/v1`：需要配置什么

配置由 `stretch_mujoco.npc.scene_config.load_scene_npc_config()` 严格读取：重复 JSON key、
未知字段、未开启的发现源、重复 NPC/spawn、无效导航参数和不完整 coverage contract 都会失败。
相对路径一律相对于配置文件本身，而不是当前工作目录。

最小结构如下；fixture 中有一份可运行的完整样例：
`models/scene_npc_configs/fixtures/minimal_scene.json`。

```json
{
  "schema": "scene_npc_config/v1",
  "scene": {
    "id": "my_scene",
    "kind": "office",
    "source_mjcf": "../../assets/office_scenes/my_scene.xml",
    "source_manifest": "../../assets/office_scenes/my_scene.json",
    "semantic_policy": "../../semantic_policies/office.json",
    "npc_catalog": "../../office_population.production.example.json",
    "navigation": {
      "surface": "office_floor",
      "planner": "collision_geometry_v1",
      "agent_radius": 0.16,
      "clearance": 0.06,
      "resolution": 0.08
    }
  },
  "semantic_registration": {
    "strict_coverage": true,
    "discover": {
      "manifest_assets": true,
      "manifest_zones": true,
      "xml_semantic_sites": true
    },
    "region_overrides": {},
    "entity_overrides": {},
    "custom_targets": {}
  },
  "population": {
    "members": [
      {
        "npc": "npc_alex_chen",
        "spawn": "point.zone.work.approach.01",
        "initial_region": "zone.work"
      }
    ]
  },
  "route_coverage": {
    "origins": "all_population_spawns",
    "targets": "all_required_navigation_points",
    "connectivity": "strongly_connected",
    "preflight": "required"
  },
  "routes": [],
  "interactions": {},
  "runtime": {"max_replans": 3, "route_failure": "fail"}
}
```

`scene.navigation` 必须与实际碰撞模型一致。当前办公室使用 `office_floor`、半径
0.16 m、额外 clearance 0.06 m、resolution 0.08 m；家庭使用
`hssd_floor_collision`、半径 0.16 m、clearance 0、resolution 0.06 m。家庭不额外膨胀
障碍物是为了不把真实可通行的窄门离散为断开的导航岛。

`population.members` 是一个场景的实例 roster：`npc` 必须存在于 `npc_catalog.npcs`，
`spawn` 必须是唯一、带 `navigation` 和 `spawn` usage 的注册点，`initial_region` 必须是
已注册实体。编译后的 population 会保留 catalog 中的 profile、embodiment、capability、
need 与 schedule，并把 spawn site/yaw 改为本场景的点位。

`routes` 是显式业务意图，而非世界坐标轨迹。每条路线需有唯一 `id`、不同的 `from`/`to`、
非空 `actions`，并选择 `fixed_contract`、`audited_dynamic` 或 `optional`。端点既可写
navigation point ID，也可写语义实体/区域 ID；后者会解析到该实体的稳定 navigation point。
运行前，trajectory profile 会基于当前碰撞几何求路径，而不会复用可能失效的 waypoint。

## 5. 注册语义点、区域和例外

编译器始终从三类来源发现信息：manifest assets、manifest zones 和 XML semantic/action
sites。三者在 `discover` 中必须全部为 `true`。发现到的实体先由 policy 分类，随后生成
可达 navigation/action point 并写入 `semantic.v2.json`；`semantic.json` 只是保守的 v1
兼容投影。

大多数新增资产只需更新 manifest 和对应 policy；不要在 Agent 或 Python 中硬编码坐标。
需要场景特化时使用以下字段：

- `custom_targets`：新增明确的自由点。ID 必须以 `point.` 开头，包含 `owner`、二维或三维
  `position`、`yaw` 和非空 `usages`。典型 usage 为 `navigation` + `spawn`、`activity` 或
  某个业务标签。给定位置必须已在主可达连通分量内；编译器会拒绝不自由或不可达的点。
- `region_overrides`：新增或修正 `room.*`/`zone.*` 区域。当前仅支持
  `geometry: {"mode": "seed_and_component", "seed": [x, y, z]}`，且必须标记
  `required: true`。这表示从种子所在的可达分量定义区域，而不是伪造房间边界。
- `entity_overrides`：纠正单一实体的 `semantic_class`、`affordances`、
  `navigation_requirement` 和 `point_bundle`。可选 `explicit_point` 或
  `action_navigation_site` 用于已有稳定站点。
- `exemption`：只允许在 `navigation_requirement: "none"` 时使用，且必须给出具体
  `reason`；active 场景目前不含任何 exemption。优先修复 manifest/XML/policy，而不是豁免。

例如，为家庭新增可达活动点可写为：

```json
"custom_targets": {
  "point.home.reading_area": {
    "owner": "room.living_room",
    "position": [-8.4, 1.2],
    "yaw": 1.57,
    "usages": ["navigation", "activity"]
  }
}
```

区域、对象和 XML action site 的注册结果应以生成的 `semantic.v2.json` 为准。每个实体含
semantic class、affordance、source XML binding，以及 `points.navigation` / `points.action`。
coverage sidecar 同时给出 discovered/classified/registered/unresolved/unbound 计数、导航参数、
主连通分量和到每个 required point 的预检结果。

## 6. 办公室和家庭的差异化配置

| 方面 | 办公室 | 家庭 |
| --- | --- | --- |
| 场景语义 | manifest category rule + office zone | category rule + exact semantic-name taxonomy + room overlay |
| 常用区域 | `zone.work`、`zone.meeting`、`zone.lounge`、`zone.snack` | `room.bathroom`、`room.bedroom`、`room.kitchen`、`room.living_room`，按该户实际发现结果注册 |
| 人口来源 | `office_population_plans.json`，每场景 2–4 人 | 每个 `home/*.json` 的 `population.members`，当前固定 3 人 |
| 自定义点 | 通常从 zone/家具自动生成 approach、seat、observation 点 | 已迁移的 household spawn/activity targets 存在每户 `custom_targets`；不要恢复旧硬编码 slot |
| 导航 | `office_floor` / 0.16 / 0.06 / 0.08 | `hssd_floor_collision` / 0.16 / 0 / 0.06 |
| 业务路线 | work→meeting、meeting→snack | living→kitchen、bedroom→living |
| 例外入口 | 优先补充 office policy 或 manifest | `home_scene_overrides.json` 用于受审查的具体实体修正 |

家庭 `room` seed 由单房间 `found_in` hint 的 medoid 投影得到，并在
`generated_scene_npc/provenance/home_room_overlays.json` 记录来源。这些是可达的语义种子，
不是人工标绘的精确房间多边形；若需要更精细的空间边界，应先扩展 schema/发现器，而不是
把未经验证的坐标散落到 Agent 代码。

## 7. 新增或修改配置的推荐流程

1. 放入或更新源 MJCF 与 manifest，确认每个有意义的 body/geom/site 有稳定名称或 manifest
   记录。不要修改 active build 内的 XML。
2. 选用或补充 `office.json` / `home.json` policy。新增一个家庭实例纠正时，将具体验证过的
   override 写入 `home_scene_overrides.json`。
3. 复制同类 `scene_npc_config/v1`，设定 source、navigation、人口、业务路线与必要的
   custom/region/entity override；新配置加入 `catalog.json`。
4. 若要加入新的 NPC identity，先在 schema-v2 NPC catalog 和其 asset manifest/
   appearance catalog 中完成可验证登记；然后才能在 `population.members` 引用该 ID。
   只复制 XML 人形不会得到合法的 NPC 身份、资产散列或动作能力。
5. 先运行 config-only audit，再编译单场景；检查 `semantic.v2` 与 coverage，修复任何
   unresolved、unbound 或 unreachable。
6. 全量 audit 成功后发布新的 active build；以新 `active_catalog.json` 的 build ID 作为
   handoff 版本。

单场景的无写入 semantic 检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/compile_scene_npc_config.py \
  stretch_mujoco/models/scene_npc_configs/office/office_02_cross_axis.json \
  --config-only
```

编译一个候选场景到非正式输出位置。`--output` 是输出文件的基名；编译器会在同目录写入
同名 XML、population、semantic、coverage、trajectory 和 receipt：

```bash
mkdir -p outputs/scene_npc_candidate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/compile_scene_npc_config.py \
  stretch_mujoco/models/scene_npc_configs/office/office_02_cross_axis.json \
  --output outputs/scene_npc_candidate/office_02_cross_axis_npc.xml
```

全量检查并原子发布 active build：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/audit_active_scene_semantics.py \
  --report stretch_mujoco/models/generated_scene_npc/coverage/active_inventory.json \
  --compile-output-root stretch_mujoco/models/generated_scene_npc/active
```

该命令先在 `active/builds/` 的临时 staging 中完成全部场景编译，只有全成功才更新
`active_catalog.json`；失败不会半发布。`tools/generate_scene_npc_resources.py` 用于从办公室
manifest、家庭 slot 计划和 taxonomy 重新生成基线配置/policy/provenance，会覆盖这些派生资源，
不应用于保留手工 scene override 的日常修改。

## 8. 从 active build 组成并运行

每个 active 场景目录内的 `*.population.json` 是 population-driven composition 的输入；
`*.trajectory_profile.json` 已被 population 引用，`*_npc.xml` 是已插入语义 site 的基础场景。
若本地 production asset projection 已就绪，可组成完整 MuJoCo 场景：

```bash
ACTIVE=stretch_mujoco/models/generated_scene_npc/active
BUILD=6ecc0ea9c8d86957031ed5ebac761c8d52a31fac9d039d54b48462a6e30896ff
POPULATION="$ACTIVE/builds/$BUILD/office_02_cross_axis/office_02_cross_axis.population.json"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m stretch_mujoco.npc.composition \
  --population "$POPULATION" \
  --output outputs/office_02_cross_axis_with_npcs.xml
```

`NpcPopulation.from_json()` 解析 population；`NpcSystem.from_population()` 会加载并校验
trajectory profile 与 scene SHA-256；`MujocoNpcActionDriver` 仅在动作与 profile 中的
source/destination/action 授权相符时采用 audited route，其他可达移动为动态重规划。每个物理
step 都必须执行 `NpcSystem.step()`，并让 Agent 消费 terminal receipt；`accepted` 不是动作
已完成。

运行时不会再按场景类型猜测导航地面：`NpcSystem` 将 profile 的 `surface`、`agent_radius`、
`clearance`、`resolution` 与 `exclude_body_roots` 原样交给每个 `LocomotionController`。因此
办公室按 `office_floor/0.16/0.06/0.08`，家庭按
`hssd_floor_collision/0.16/0/0.06` 建立同一套碰撞网格；重规划和动态 NPC 占用检查也使用
该契约。profile 已绑定但指定 surface 不存在、无法建图或无法找到路线时，控制器以
`navigation_surface_missing:*` 或 `route_unavailable` 终止，绝不回退为直线移动。仅没有
trajectory profile 的历史最小单元测试 fixture 保留旧 `office_floor` fallback，不能用于
办公室或家庭 production scene。

MuJoCo 已编译的 `MjModel` 不能热插入 body。因此人数、spawn、appearance 或 scene 改动后
必须重新编译并重新加载，不能在运行中篡改 XML/JSON 或直接改 controller 状态。

## 9. 修改后的检查清单

至少执行与修改范围相符的检查：

```bash
# 20 个配置的严格发现/分类检查（无 active 发布）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python tools/audit_active_scene_semantics.py \
  --report /tmp/scene_npc_inventory.json

# 本轮统一场景编译的回归集
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_generated_home_npc_preprocessing.py \
  tests/test_generated_office_npc_preprocessing.py \
  tests/test_npc_composition.py \
  tests/test_scene_npc_compiler.py \
  tests/test_npc_trajectory_profile.py
```

若改动 NPC asset、appearance 或 bundle，还应运行相应 population 的
`validate_npc_assets`，并确认 manifest、appearance catalog、所有受引用文件 SHA-256 和
capability/clip 合约均一致。若改动 MJCF 碰撞、点位、surface、半径或 clearance，必须重新
运行全量发布命令；旧的 trajectory profile 或 receipt 不能证明新几何仍可达。

## 10. 常见错误

| 现象 | 优先处理方式 |
| --- | --- |
| `unknown_root_fields`、`duplicate_key` 或 schema 错误 | 严格按第 4 节字段写 JSON；不要试图加入 loader 未支持的扩展字段。 |
| `npc_catalog_member_missing` | 先在 NPC catalog/asset manifest 中登记身份，再在 scene roster 引用。 |
| route endpoint unbound / no navigation point | 修复 manifest/XML binding 或 policy/override；不要改 Agent 中的坐标。 |
| unreachable / 非强连通 | 检查 surface、机器人排除体、NPC 半径、clearance、障碍碰撞和 spawn；通过实际碰撞几何修复。 |
| 生成的 XML 找不到 include/asset | 使用同一 active scene 目录的完整产物；不要把文件单独复制到其他目录。 |
| 修改生成 JSON 后下次消失 | 这是预期行为；将改动移动到 `scene_npc_configs`、policy、manifest 或 roster 资源。 |
| production asset 缺失 | 明确使用 preview 示例做测试，或补齐本地受限资产投影；不要把 preview 伪装为 production。 |
## Active showcase foundation (partial evidence)

The reusable runner is `tools/render_active_scene_showcase.py`. Its source scenarios live in
`aaa_workspace/showcases/office_02_day_in_the_life.json` and
`aaa_workspace/showcases/home_04_household_assistance.json`. The runner resolves the active
catalog, validates every declared route/action against the active trajectory profile, delegates
movement to the profile-bound `NpcSystem`, and writes a video plus a JSON receipt with profile
preflight and collision trace audit. The runner currently executes only its selected movement
phase; every other phase is marked `not_executed`, so this is partial evidence, not a complete
demo. Conversation and robot handover phases now use the active population's explicit capability
projection: unsupported source capabilities remain fail-safe with their configured reason, while
supported capabilities are reported ready for their corresponding executor.

### 10.1 Source scene traffic and interaction capabilities

Every office/home source scene config declares a `traffic` contract and exactly two interaction
capability outcomes under `interactions`: `conversation` and `robot_handover`. Traffic uses
`sequential_route_reservation`, enables dynamic-obstacle handling, and sets
`no_direct_fallback: true`; a profile-bound controller must wait/replan or fail safely.

Each capability is either `supported` or `unsupported`. Unsupported entries require a concrete
`reason`. Supported conversation entries require at least two named role sites. Supported robot
handover entries additionally require two role sites, a manifest-backed object, and a source-MJCF
robot approach site. The compiler validates these bindings against source resources and rejects
claims that cannot be proven.

The compiler projects the source declarations into the generated population as
`traffic_policy` and `interaction_capabilities`. Showcase consumers read only these projections;
they must not infer support from scene kind or from an empty interaction map. Current source
scenes explicitly mark both capabilities unsupported because no complete source-backed contract
has been registered yet. To add support, register real named sites in source MJCF, bind the
object in the source manifest, update the matching source config, then run strict compilation and
preflight. Never edit generated active JSON directly.
