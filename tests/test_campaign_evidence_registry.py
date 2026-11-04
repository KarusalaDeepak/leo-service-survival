from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_campaign_registry_is_fail_closed_and_has_unique_groups():
    registry = pd.read_csv(ROOT / "configs" / "campaign_evidence_registry.csv")
    assert registry["campaign_id"].is_unique
    assert registry["acquisition_family"].nunique() >= 3

    for column in (
        "synchronized_opportunities",
        "paired_outcome_available",
        "multi_access_gate_eligible",
    ):
        registry[column] = registry[column].map({True: True, False: False})
        assert registry[column].notna().all()

    admitted = registry.loc[registry["multi_access_gate_eligible"]]
    assert admitted["campaign_id"].tolist() == ["commect_denmark_drive"]
    assert admitted["synchronized_opportunities"].all()
    assert admitted["paired_outcome_available"].all()
    assert admitted["exclusion_or_scope_reason"].str.len().gt(0).all()

    fjordlink = registry.set_index("campaign_id").loc[
        "fjordlink_kiel_fjord_2024_2025"
    ]
    assert not bool(fjordlink["multi_access_gate_eligible"])
    assert "zero" in fjordlink["exclusion_or_scope_reason"].lower()

    for campaign_id in (
        "nuwins_alaska_202406",
        "nuwins_hawaii_202408",
        "nuwins_mainland_202411",
    ):
        nuwins = registry.set_index("campaign_id").loc[campaign_id]
        assert not bool(nuwins["multi_access_gate_eligible"])
        assert not bool(nuwins["synchronized_opportunities"])
        assert not bool(nuwins["paired_outcome_available"])


def test_campaign_count_is_not_used_as_row_count_or_family_count():
    registry = pd.read_csv(ROOT / "configs" / "campaign_evidence_registry.csv")
    admitted = registry.loc[registry["multi_access_gate_eligible"]]
    assert admitted["campaign_id"].nunique() == len(admitted)
    assert admitted["acquisition_family"].nunique() <= len(admitted)
    assert "records" not in " ".join(registry.columns).lower()
