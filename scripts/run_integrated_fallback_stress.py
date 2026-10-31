#!/usr/bin/env python3
"""Run the deterministic synthetic EOS input-guard fallback integration test."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from leo_service_survival.fallback_stress import (
    run_synthetic_integrated_fallback,
    write_integrated_fallback_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "eos_integrated_fallback_synthetic_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the real EOS gate-to-reactive-fallback path with a deliberately "
            "unsupported synthetic input. This never produces field evidence."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026102801)
    args = parser.parse_args()

    artifacts = run_synthetic_integrated_fallback(seed=args.seed)
    output = write_integrated_fallback_artifacts(artifacts, args.output)
    report = {
        "output": str(output.resolve()),
        "evidence_tier": artifacts.claims_ledger["evidence_tier"],
        "counts": artifacts.claims_ledger["counts"],
        "checks": artifacts.claims_ledger["checks"],
        "sha256sums_sha256": sha256((output / "SHA256SUMS").read_bytes()).hexdigest(),
        "claim_boundary": artifacts.claims_ledger["claim_boundary"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
