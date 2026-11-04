#!/usr/bin/env python3
"""Run the deterministic FAKE_DRIVER_TEST rehearsal of all seven O4 steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.controlled_dry_run import (  # noqa: E402
    run_controlled_seven_step_dry_run,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "prospective_trial" / "controlled_dry_run",
    )
    parser.add_argument(
        "--acknowledge-fake-driver-test",
        action="store_true",
        help="Required acknowledgement that no artifact is field evidence.",
    )
    args = parser.parse_args()
    if not args.acknowledge_fake_driver_test:
        raise RuntimeError("pass --acknowledge-fake-driver-test")
    result = run_controlled_seven_step_dry_run(args.output, package_root=ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
