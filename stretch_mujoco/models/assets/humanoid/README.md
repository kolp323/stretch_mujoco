# NPC humanoid assets / NPC 人形资产

## 中文

本目录将可再分发的 preview fixture 与本地 production payload 分离。这一区分是
NPC 资产契约的一部分：production 配置不得静默回退到 preview 网格，私有或生成的
资源也不得提交到仓库。

| 路径 | 所有权与用途 | 版本控制策略 |
| --- | --- | --- |
| `cesium_man.*`、`rigged_figure.*` | 供 `npc_assets.example.json` 和示例使用的小型 CC BY preview fixture。 | 跟踪；manifest 中的 SHA-256 是完整性契约。 |
| `sources/` | 已审核 source archive，按 `npc/accessories`、`npc/hair`、`npc/textures` 组织。 | 本地共享库；只跟踪 README。 |
| `private/` | 已授权的 SMPL-X 参数，以及 AMASS 输入和 receipt。 | 仅本地；只跟踪 README。 |
| `generated/` | 派生 OBJ 帧、纹理、anchor、manifest、缩略图和 runtime projection。 | 仅本地；只跟踪 README。 |

### Preview fixture

`cesium_man.glb`、`cesium_man_idle.obj` 和 `cesium_man.png` 构成一个可再分发的
preview bundle。`npc_assets.example.json` 固定 OBJ 和 PNG 的 SHA-256，因此干净克隆
无需下载受限资产也能验证和编译 preview population。GLB 是 Khronos glTF Sample
Assets 中的 CesiumMan 动画样例（Copyright 2017 Cesium，CC BY 4.0；适用其商标条款）：

https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan

`cesium_man_idle.obj` 是从第 36 帧转换得到的 MuJoCo 兼容 OBJ，并由
`office_scene.xml` 缩放至约 1.72 m。`cesium_man.png` 是对应的 1024×1024 MuJoCo
diffuse texture。变更任一文件的 bytes 时，必须同步更新
`models/npc_assets.example.json` 中的 SHA-256。

较小的 RiggedFigure 文件仍作为同一 Khronos collection 的 CC BY 4.0 skeleton
reference 保留。两个 GLB 都保留原始 skeleton 和 animation；当前 runtime 使用的是
确定性的 OBJ mesh sequence。

### Production 资产生命周期

1. 将带有 provenance 和 licence evidence 的已审核 archive 放入 `sources/`。
2. 将受限的 SMPL-X/AMASS 输入放入 `private/`。
3. 按 `private/README.md` 的命令在 `generated/` 中生成并验证 frames、textures、
   anchors、receipts 和 manifests。
4. production population 只能引用生成后的 manifest；构建 scene 前执行
   `uv run validate_npc_assets --population <population.json>`。

完整的配置与验证流程见 [NPC system guide](../../../../docs/npc_system.md)。

## English

This directory separates redistributable preview fixtures from local production
payloads. This distinction is part of the NPC asset contract: a production
configuration must never silently fall back to a preview mesh, and private or
generated payloads must not be committed.

| Path | Ownership and purpose | Version-control policy |
| --- | --- | --- |
| `cesium_man.*`, `rigged_figure.*` | Small CC BY preview fixtures used by `npc_assets.example.json` and examples. | Tracked; manifest SHA-256 values are the integrity contract. |
| `sources/` | Approved source archives organised by `npc/accessories`, `npc/hair`, and `npc/textures`. | Local shared store; only its README is tracked. |
| `private/` | Licensed SMPL-X parameters and AMASS inputs/receipts. | Local only; only its README is tracked. |
| `generated/` | Derived OBJ frames, textures, anchors, manifests, thumbnails, and runtime projections. | Local only; only its README is tracked. |

### Preview fixture

`cesium_man.glb`, `cesium_man_idle.obj`, and `cesium_man.png` form one
redistributable preview bundle. `npc_assets.example.json` pins the OBJ and PNG
SHA-256 values, so a clean clone can validate and compile the preview
population without downloading licensed assets. The GLB is the animated
CesiumMan sample from Khronos glTF Sample Assets (Copyright 2017 Cesium,
CC BY 4.0; its trademark terms apply):

https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan

`cesium_man_idle.obj` is frame 36 converted to a MuJoCo-compatible OBJ and
scaled by `office_scene.xml` to roughly 1.72 m. `cesium_man.png` is the
corresponding 1024×1024 MuJoCo diffuse texture. Do not change either file's
bytes without updating the SHA-256 values in `models/npc_assets.example.json`.

The smaller RiggedFigure files remain a CC BY 4.0 skeleton reference from the
same Khronos collection. Both GLBs retain their original skeletons and
animations; the current runtime uses deterministic OBJ mesh sequences instead.

### Production asset lifecycle

1. Keep a reviewed archive with provenance and licence evidence in `sources/`.
2. Keep restricted SMPL-X/AMASS inputs in `private/`.
3. Generate and validate frames, textures, anchors, receipts, and manifests in
   `generated/` using the commands in `private/README.md`.
4. Reference only the generated manifest from a production population and run
   `uv run validate_npc_assets --population <population.json>` before building
   a scene.

See the [NPC system guide](../../../../docs/npc_system.md) for the full
configuration and verification workflow.
