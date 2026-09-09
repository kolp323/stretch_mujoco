# Generated SMPL-X assets

This directory is populated locally by `prepare_smplx_npc` and
`bake_smplx_animations`. Generated meshes remain subject to the SMPL-X license
and are ignored by Git. See `../private/README.md` for generation commands.

The animation baker now emits a validated NPC asset `manifest.json` containing
the formal `idle`, `walk`, `sit`, `work`, and `eat` clips, animation markers,
and SHA-256 hashes. Use it with `office_population.production.example.json`;
the checked-in `office_population.json` deliberately remains a redistributable
CesiumMan preview configuration.
