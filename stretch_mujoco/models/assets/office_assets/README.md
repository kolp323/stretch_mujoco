# Office Assets Catalog

Extracted from HSSD Habitat scene `108294417_176709879`.

| Category | Asset | File |
|----------|-------|------|
| chairs | Bucket Seat Dining Chair Grey Velvet | [dining_chair.xml](chairs/dining_chair.xml) |
| chairs | Hooker Furniture Katherine Home Office Chair | [home_office_chair.xml](chairs/home_office_chair.xml) |
| chairs | Cornell Swivel Office Chair | [office_chair.xml](chairs/office_chair.xml) |
| chairs | Stance Chair | [stance_chair.xml](chairs/stance_chair.xml) |
| desks | Smartstudy Modular Adjustable ELEM | [adjustable_desk.xml](desks/adjustable_desk.xml) |
| desks | CB Desk 2400 Dressed | [cb_desk_2400.xml](desks/cb_desk_2400.xml) |
| desks | Custom Oak Sq Table Cranbrook Leg | [oak_square_table.xml](desks/oak_square_table.xml) |
| desks | CB Reception Desk Curved | [reception_desk.xml](desks/reception_desk.xml) |
| displays | Apple iMAC Core 2 Duo 24 | [imac_24.xml](displays/imac_24.xml) |
| displays | APPLE iMac 5K 27 | [imac_27.xml](displays/imac_27.xml) |
| meeting_table | SmartStudy Sit-Stand Teaming Table | [teaming_table.xml](meeting_table/teaming_table.xml) |
| whiteboard | TW Dry Wipe Module | [dry_wipe_module.xml](whiteboard/dry_wipe_module.xml) |

## Restored legacy HSSD library

The original 39 textured assets are restored alongside the newer scene-extracted MJCF files.
They retain their original category directories under `furniture/`, `electronics/`, `lighting/`,
`props/`, and `decoration/`. Their combined asset registry is
[`mjcf/office_assets.xml`](mjcf/office_assets.xml), and the exact restoration manifest is
[`catalog/legacy_restored.json`](catalog/legacy_restored.json).

Set `STRETCH_MUJOCO_HSSD_ROOT` to an HSSD checkout, then rebuild the legacy library with:

```bash
uv run python tools/restore_legacy_office_assets.py
```
