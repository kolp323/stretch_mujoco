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
