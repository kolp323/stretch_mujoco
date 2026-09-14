#!/usr/bin/env python3
"""Strict, simulator-free validation for a v1 office control sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stretch_mujoco.agents.control_sequences import (
    ControlSequenceError,
    SequenceCompiler,
    StrictSequenceLoader,
)
from stretch_mujoco.agents.control_sequences.preflight import discover_capabilities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="control-sequence YAML")
    parser.add_argument("--control-mode", choices=("yaml", "llm"))
    parser.add_argument("--strict", action="store_true", help="kept for explicit CI invocation")
    args = parser.parse_args(argv)
    try:
        sequence = StrictSequenceLoader().load(args.sequence)
        capabilities = discover_capabilities(sequence)
        compiled = SequenceCompiler(capabilities).compile(sequence, args.control_mode)
    except ControlSequenceError as error:
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "sequence_id": sequence.sequence_id,
                "effective_control_mode": compiled.control_mode.value,
                "sequence_sha256": sequence.source_sha256,
                "compiled_plan_sha256": compiled.plan_sha256,
                "covered_actions": sorted(compiled.covered_actions),
                "site_count": len(capabilities.site_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
