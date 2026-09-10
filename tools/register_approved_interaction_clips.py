#!/usr/bin/env python3
"""Register reviewed interaction OBJ sequences in the production asset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from stretch_mujoco.humanoid.smplx_animation_baker import register_approved_interaction_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-clips", type=Path, required=True)
    args = parser.parse_args()
    manifest = register_approved_interaction_clips(args.manifest, args.approved_clips)
    bundle_id = next(iter(manifest["bundles"]))
    clip_count = len(manifest["bundles"][bundle_id]["clips"])
    print(f"Registered {clip_count} production clips in '{args.manifest}'")


if __name__ == "__main__":
    main()
