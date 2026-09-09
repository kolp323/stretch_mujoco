# Generated SMPL-X assets

This directory is populated locally by `prepare_smplx_npc` and
`bake_smplx_animations`. Generated meshes remain subject to the SMPL-X license
and are ignored by Git. See `../private/README.md` for generation commands.

The animation baker accepts only locally selected, provenance-backed motion
inputs. It emits a restricted `manifest.json` containing the complete office
clip contract, animation markers, and SHA-256 hashes. Missing clips reject
generation; neither preview frames nor synthetic poses are substituted. Use it
with `office_population.production.example.json`; the checked-in
`office_population.json` deliberately remains a redistributable CesiumMan
preview configuration.
