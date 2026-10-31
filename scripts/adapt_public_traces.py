#!/usr/bin/env python3
"""Create bounded, auditable canonical tables from acquired public traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leo_service_survival.lens_adapter import (  # noqa: E402
    align_lens_irtt_to_dish,
    build_lens_connectivity_episodes,
    load_lens_dish_status,
    load_lens_irtt,
    load_lens_ping,
)
from leo_service_survival.nuwins_adapter import (  # noqa: E402
    MainlandAlignmentConfig,
    align_nuwins_mainland_pair,
    load_nuwins_pre_aligned_panel,
)
from leo_service_survival.sat_in_view_adapter import (  # noqa: E402
    load_sat_in_view_matches,
)


def _output_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_manifest(output: Path, command: str, arguments: dict, rows: dict) -> None:
    payload = {
        "adapter_command": command,
        "arguments": arguments,
        "output_rows": rows,
        "claim_boundary": (
            "Canonicalized observational or measured-trace evidence only; adapters "
            "do not create serving-satellite, beam, gateway, RF, geometry, policy-action, "
            "or operational EO-completion evidence."
        ),
    }
    (output / "adapter_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _nuwins_pre_aligned(args: argparse.Namespace) -> None:
    output = _output_directory(args.output_dir)
    observations = load_nuwins_pre_aligned_panel(
        args.input,
        campaign_id=args.campaign_id,
        panel_kind=args.panel_kind,
        max_rows=args.max_rows,
    )
    observations.to_csv(output / "candidate_path_observations.csv", index=False)
    _write_manifest(
        output,
        "nuwins-pre-aligned",
        vars(args),
        {"candidate_path_observations": len(observations)},
    )


def _nuwins_mainland(args: argparse.Namespace) -> None:
    output = _output_directory(args.output_dir)
    result = align_nuwins_mainland_pair(
        args.anchor,
        args.candidate,
        campaign_id=args.campaign_id,
        anchor_operator=args.anchor_operator,
        candidate_operator=args.candidate_operator,
        panel_kind=args.panel_kind,
        config=MainlandAlignmentConfig(
            run_tolerance_ms=args.run_tolerance_ms,
            sample_tolerance_ms=args.sample_tolerance_ms,
            max_anchor_rows=args.max_anchor_rows,
            max_candidate_rows=args.max_candidate_rows,
        ),
    )
    result.observations.to_csv(
        output / "candidate_path_observations.csv", index=False
    )
    result.audit.to_csv(output / "alignment_audit.csv", index=False)
    _write_manifest(
        output,
        "nuwins-mainland",
        vars(args),
        {
            "candidate_path_observations": len(result.observations),
            "alignment_audit": len(result.audit),
            "eligible_anchor_rows": int(result.audit["alignment_eligible"].sum()),
        },
    )


def _lens(args: argparse.Namespace) -> None:
    output = _output_directory(args.output_dir)
    if args.probe_kind == "irtt":
        probes = load_lens_irtt(
            args.probes,
            max_records=args.max_probe_records,
            campaign_id=args.campaign_id,
            node_id=args.node_id,
        )
    else:
        probes = load_lens_ping(
            args.probes,
            max_records=args.max_probe_records,
            max_members=args.max_probe_members,
            member_glob=args.member_glob,
            start=args.start,
            end=args.end,
            campaign_id=args.campaign_id,
            node_id=args.node_id,
        )
    telemetry = load_lens_dish_status(
        args.dish_status,
        start=args.start,
        end=args.end,
        max_records=args.max_telemetry_records,
        campaign_id=args.campaign_id,
    )
    aligned = align_lens_irtt_to_dish(
        probes, telemetry, max_telemetry_age_s=args.max_telemetry_age_s
    )
    episodes = build_lens_connectivity_episodes(
        aligned, maximum_contiguous_gap_ms=args.maximum_contiguous_gap_ms
    )
    probes.to_csv(output / "probe_observations.csv", index=False)
    telemetry.to_csv(output / "dish_telemetry.csv", index=False)
    aligned.to_csv(output / "aligned_probe_telemetry.csv", index=False)
    episodes.to_csv(output / "connectivity_episodes.csv", index=False)
    _write_manifest(
        output,
        "lens",
        vars(args),
        {
            "probe_observations": len(probes),
            "dish_telemetry": len(telemetry),
            "aligned_probe_telemetry": len(aligned),
            "telemetry_aligned": int(aligned["telemetry_aligned"].sum()),
            "connectivity_episodes": len(episodes),
        },
    )


def _sat_in_view(args: argparse.Namespace) -> None:
    output = _output_directory(args.output_dir)
    matches = load_sat_in_view_matches(
        args.input,
        max_rows=args.max_rows,
        maximum_transition_gap_s=args.maximum_transition_gap_s,
    )
    matches.to_csv(output / "inferred_satellite_matches.csv", index=False)
    _write_manifest(
        output,
        "sat-in-view",
        vars(args),
        {
            "inferred_satellite_matches": len(matches),
            "inferred_contiguous_identity_changes": int(
                matches["inferred_handover_event"].sum()
            ),
            "identity_changes_unresolved_across_gaps": int(
                matches["identity_change_unresolved_across_gap"].sum()
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    paired = subparsers.add_parser("nuwins-pre-aligned")
    paired.add_argument("input")
    paired.add_argument("output_dir")
    paired.add_argument("--campaign-id", required=True)
    paired.add_argument(
        "--panel-kind", choices=("downlink_throughput", "rtt"), required=True
    )
    paired.add_argument("--max-rows", type=int)
    paired.set_defaults(function=_nuwins_pre_aligned)

    mainland = subparsers.add_parser("nuwins-mainland")
    mainland.add_argument("anchor")
    mainland.add_argument("candidate")
    mainland.add_argument("output_dir")
    mainland.add_argument("--campaign-id", default="mainland_202411")
    mainland.add_argument("--anchor-operator", required=True)
    mainland.add_argument("--candidate-operator", required=True)
    mainland.add_argument(
        "--panel-kind", choices=("downlink_throughput", "rtt"), required=True
    )
    mainland.add_argument("--run-tolerance-ms", type=float, default=5_000.0)
    mainland.add_argument("--sample-tolerance-ms", type=float)
    mainland.add_argument("--max-anchor-rows", type=int)
    mainland.add_argument("--max-candidate-rows", type=int)
    mainland.set_defaults(function=_nuwins_mainland)

    lens = subparsers.add_parser("lens")
    lens.add_argument("probes")
    lens.add_argument("dish_status")
    lens.add_argument("output_dir")
    lens.add_argument("--probe-kind", choices=("irtt", "ping"), required=True)
    lens.add_argument("--campaign-id")
    lens.add_argument("--node-id")
    lens.add_argument("--start")
    lens.add_argument("--end")
    lens.add_argument("--member-glob")
    lens.add_argument("--max-probe-records", type=int, default=100_000)
    lens.add_argument("--max-probe-members", type=int, default=1)
    lens.add_argument("--max-telemetry-records", type=int, default=10_000)
    lens.add_argument("--max-telemetry-age-s", type=float, default=75.0)
    lens.add_argument("--maximum-contiguous-gap-ms", type=float, default=50.0)
    lens.set_defaults(function=_lens)

    satellite = subparsers.add_parser("sat-in-view")
    satellite.add_argument("input")
    satellite.add_argument("output_dir")
    satellite.add_argument("--max-rows", type=int, default=100_000)
    satellite.add_argument("--maximum-transition-gap-s", type=float, default=2.0)
    satellite.set_defaults(function=_sat_in_view)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
