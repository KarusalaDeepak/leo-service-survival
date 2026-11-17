#!/usr/bin/env python3
"""Hash each public result/figure and map it to one producer entry point."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/result_to_script_manifest.csv"


# These files are useful for private diagnosis and exact local reruns, but they
# are row-level decision traces rather than publication-safe aggregate results.
# Keep the denylist explicit and fail closed: the public manifest must never
# make them look like cleared aggregate artifacts merely because they live
# below an otherwise public result directory.
PUBLIC_MANIFEST_EXCLUSIONS = {
    "results/eos_mechanism_ablation_v3/action_set_audit.csv",
    "results/eos_mechanism_ablation_v3/candidate_scores.csv",
    "results/eos_mechanism_ablation_v3/task_results.csv",
}


RULES = (
    (
        "results/parameter_revision_r5/",
        "scripts/build_parameter_revision_evidence.py",
        "main",
        "explicit aggregate-only export of separate endpoint, development and synthetic parameter audits",
        "aggregate_public",
    ),
    (
        "results/review_audit/ipcw_boundary_",
        "scripts/audit_ipcw_boundaries.py",
        "main",
        "hand-calculated and implementation IPCW boundary fixtures",
        "aggregate_public",
    ),
    (
        "results/reviewer_prediction_release/",
        "scripts/build_prediction_release_artifacts.py",
        "main",
        "WetLinks derived held-out predictions and scores",
        "wetlinks_derivative_cc_by_sa_4_0",
    ),
    (
        "results/review_audit/",
        "scripts/build_manuscript_audits.py",
        "main",
        "focused manuscript audit inputs",
        "aggregate_public",
    ),
    (
        "results/early_stopping_sensitivity/",
        "scripts/run_early_stopping_sensitivity.py",
        "main",
        "WetLinks measured episodes",
        "aggregate_public",
    ),
    (
        "results/eos_mechanism/",
        "scripts/run_eos_mechanism_ablation.py",
        "main",
        "controlled synthetic EOS mechanism study",
        "aggregate_public",
    ),
    (
        "results/eos_mechanism_ablation_v3/",
        "scripts/run_eos_mechanism_ablation.py",
        "main",
        "controlled synthetic EOS mechanism study with development-only parameter selection",
        "aggregate_public",
    ),
    (
        "results/gate_logic/",
        "scripts/run_gate_stress.py",
        "main",
        "controlled synthetic gate stress",
        "aggregate_public",
    ),
    (
        "results/gate_parameter_governance_v1/",
        "scripts/run_gate_stress.py",
        "main",
        "controlled synthetic gate stress and prospective campaign precision planning",
        "aggregate_public",
    ),
    (
        "results/matched_telemetry/",
        "scripts/run_matched_telemetry_audit.py",
        "main",
        "bounded matched-telemetry audit",
        "aggregate_public",
    ),
    (
        "results/manuscript/",
        "scripts/run_all.py",
        "main",
        "mixed measured, replay, and synthetic manuscript evaluation",
        "aggregate_public",
    ),
    (
        "figures/event_censoring.png",
        "scripts/build_event_censoring_figure.py",
        "main",
        "released aggregate tables",
        "figure_public",
    ),
    (
        "figures/decision_contrasts.png",
        "scripts/build_decision_contrasts_figure.py",
        "main",
        "released aggregate tables",
        "figure_public",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(relative_path: str) -> tuple[str, str, str, str]:
    if relative_path.endswith("/README.md"):
        return (
            "manual_documentation",
            "not_applicable",
            "human-readable artifact documentation",
            (
                "wetlinks_derivative_cc_by_sa_4_0"
                if relative_path.startswith("results/reviewer_prediction_release/")
                else "documentation_public"
            ),
        )
    matches = [rule for rule in RULES if relative_path.startswith(rule[0])]
    if not matches:
        raise ValueError(f"no producer rule for {relative_path}")
    longest = max(len(rule[0]) for rule in matches)
    selected = [rule for rule in matches if len(rule[0]) == longest]
    if len(selected) != 1:
        raise ValueError(
            f"ambiguous producer rules for {relative_path}: {len(selected)}"
        )
    _prefix, script, function, inputs, redistribution = selected[0]
    return script, function, inputs, redistribution


def main() -> None:
    paths = sorted(
        path
        for folder in (ROOT / "results", ROOT / "figures")
        for path in folder.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and path.name != ".DS_Store"
        and "__pycache__" not in path.parts
        and path.relative_to(ROOT).as_posix() not in PUBLIC_MANIFEST_EXCLUSIONS
    )
    rows = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        producer, function, inputs, redistribution = _mapping(relative)
        rows.append(
            {
                "artifact_path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "producer_entrypoint": producer,
                "producer_function": function,
                "input_or_evidence_class": inputs,
                "redistribution_class": redistribution,
                "model_object_required": False,
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

