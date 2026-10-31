import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results" / "matched_telemetry"


def test_matched_telemetry_audit_preserves_evidence_boundary():
    summary = pd.read_csv(OUTPUT / "matched_telemetry_summary.csv")
    status = json.loads((OUTPUT / "criterion_status.json").read_text())
    oneweb = summary.loc[summary["source"].str.contains("OneWeb")].iloc[0]

    assert int(oneweb["rows"]) == 599
    assert int(oneweb["matched_rtt_rows"]) == 594
    assert int(oneweb["observed_or_inferred_transitions"]) == 2
    assert int(oneweb["transition_windows_with_complete_5s_context"]) == 2
    assert status["satisfied"] is False
    assert "beam and gateway" in status["reason"]
