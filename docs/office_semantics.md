# Office Semantic World

The office semantic layer is defined in
`stretch_mujoco/models/office_semantics.json` and loaded with `SemanticWorld`.
MuJoCo remains authoritative for geometry and poses; the semantic graph is
authoritative for identity, ownership, permissions, requests, and task state.

## Relation Direction

Relations are stored as `(subject, relation, object)` triples:

```text
document_report ON storage_cabinet (robot dispatch tray)
document_report BELONGS_TO employee_01
employee_01 HOLDS document_report
document_report REQUESTED_BY employee_01
```

`INSIDE` and `ON` are mutually exclusive locations when changed through
`set_location()`. Confidential objects require an `ALLOWED_FOR` relation for
`can_access()` to return true.

## Agent Usage

```python
from stretch_mujoco.semantics import RelationType

world = sim.semantic_world
requests = world.pending_requests("employee_01")

if world.can_access("stretch_3", "document_report"):
    world.add_relation("stretch_3", RelationType.HOLDS, "document_report")
    world.remove_relation(
        "document_report", RelationType.ON, "storage_cabinet"
    )
```

After placing an object, update its logical location atomically:

```python
world.set_location("document_report", RelationType.ON, "meeting_table")
```

`sim.pull_semantic_state()` provides a 5 Hz process-safe snapshot containing
world-space poses for every bound object and interaction point. Use these poses
as navigation, manipulation, handover, or animation targets.

## Automatic NPC Navigation Grid

`OfficeNavigationMesh` derives its walkable bounds from `office_floor` and
rasterizes MuJoCo collision geoms into an 8 cm grid. Obstacles are inflated by the
NPC's 25 cm horizontal safety radius. An 8-neighbor A* search prevents diagonal
corner cutting, then an analytic line-segment/AABB test smooths the route without
allowing it to graze furniture.

Only semantic human targets such as `meeting_human_stand_site` and
`snack_human_stand_site` remain authored. Intermediate waypoints are generated
automatically, so moving furniture and rebuilding the model changes the route
without editing a waypoint list. Chair use plans to a collision-free approach point
before handing the final seated alignment to the sit animation.
