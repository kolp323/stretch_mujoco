# Generated simplified home scenes

Ten scenes generated from ten different HSSD scene IDs. The script uses the
HSSD `scenes-uncluttered` split, converts each stage/object to MuJoCo, and adds
a Hello Robot Stretch include. Converted meshes and textures are stored in
`_hssd_cache/`.

```bash
python examples/generated_home_scene.py --scene 1
```

Regenerate with a different dataset location using:

```bash
python tools/generate_home_scenes.py --hssd-root /path/to/hssd-hab
```

Prepare these scenes for the population-driven NPC runtime with:

```bash
.venv/bin/python tools/preprocess_generated_home_npcs.py
```

`npc_slot_overrides.json` is the reviewed, schema-v2 site plan for each home.
Its `sites` mapping owns arbitrary named coordinates, while `roster` chooses
which template NPCs spawn at which named sites; thus the number of NPCs is the
length of `roster`, not a slot index in Python. `demo` is optional and only
selects an activity site plus two named roster members for the home-video
renderer. Add named sites and roster entries there when onboarding another
scene rather than encoding a scene-specific position in Python.

The derived `*_npc.xml` files and their population/semantic/receipt sidecars
are documented in `docs/generated_home_npc.md`.  The converted HSSD cache is
required for MuJoCo compilation and remains outside Git.
