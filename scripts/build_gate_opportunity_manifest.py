#!/usr/bin/env python3
"""Verify and inventory Step-2 hash-locked raw gate-opportunity tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.gate_freeze import (  # noqa: E402
    CONFIRMATORY_LIVE,
    CONTROLLED_FIXTURE,
    TrialFreezeManifest,
    build_gate_opportunity_manifest,
)
from leo_service_survival.prospective_release import load_signing_key  # noqa: E402


def _table_argument(value: str) -> tuple[str, Path]:
    campaign, separator, path = value.partition("=")
    if not separator or not campaign or not path:
        raise argparse.ArgumentTypeError("table must be CAMPAIGN_ID=/path/to/table.csv")
    return campaign, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-freeze-manifest", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument(
        "--execution-mode",
        choices=(CONFIRMATORY_LIVE, CONTROLLED_FIXTURE),
        required=True,
    )
    parser.add_argument(
        "--table",
        type=_table_argument,
        action="append",
        required=True,
        help="Repeat once per gate campaign: CAMPAIGN_ID=/path/to/table.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite opportunity manifest: {args.output}")
    table_paths = dict(args.table)
    if len(table_paths) != len(args.table):
        raise SystemExit("duplicate campaign id in --table arguments")
    manifest = TrialFreezeManifest.from_dict(
        json.loads(args.trial_freeze_manifest.read_text(encoding="utf-8"))
    )
    value = build_gate_opportunity_manifest(
        freeze_manifest=manifest,
        table_paths=table_paths,
        signing_key=load_signing_key(args.signing_key),
        execution_mode=args.execution_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evidence_status": value["evidence_status"],
                "tables": len(value["tables"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

