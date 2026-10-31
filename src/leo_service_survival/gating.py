"""Frozen evidence gate and online shift monitors for LEO decisions.

The release-gate comparison and the construction of its statistical inputs are
two different validation levels.  :class:`EvidenceGate` tests the former.  A
caller must label whether the supplied summaries are a logic-test fixture or
bounds generated from campaign observations; the gate reports that distinction
without promoting a supplied fixture to statistical authorization evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.stats import beta


GATE_LOGIC_FIXTURE = "gate_logic_fixture"
CAMPAIGN_DERIVED_STATISTICAL_BOUNDS = "campaign_derived_statistical_bounds"
_RECOGNIZED_EVIDENCE_BASES = {
    GATE_LOGIC_FIXTURE,
    CAMPAIGN_DERIVED_STATISTICAL_BOUNDS,
}


def _finite_numeric(value: object) -> tuple[bool, float]:
    """Return a strict numeric/finite check without treating booleans as data."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False, np.nan
    numeric = float(value)
    return bool(np.isfinite(numeric)), numeric


def _same_numeric_value(left: object, right: object) -> bool:
    left_valid, left_value = _finite_numeric(left)
    right_valid, right_value = _finite_numeric(right)
    return bool(left_valid and right_valid and left_value == right_value)


@dataclass(frozen=True, init=False)
class GateCriteria:
    """Thresholds fixed before a locked campaign is inspected.

    ``minimum_safety_effect_lower`` remains an accepted constructor alias for
    archived scripts.  The canonical name states the contrast direction and
    makes clear that the lower bound applies to a declared endpoint.
    """

    minimum_independent_campaigns: int
    minimum_feature_coverage: float
    maximum_ood_fraction: float
    maximum_calibration_gap_upper: float
    minimum_candidate_minus_baseline_endpoint_effect_lower: float
    maximum_control_loop_p95_ms: float

    def __init__(
        self,
        minimum_independent_campaigns: int = 2,
        minimum_feature_coverage: float = 0.95,
        maximum_ood_fraction: float = 0.10,
        maximum_calibration_gap_upper: float = 0.05,
        minimum_candidate_minus_baseline_endpoint_effect_lower: float | None = None,
        maximum_control_loop_p95_ms: float = 100.0,
        *,
        minimum_safety_effect_lower: float | None = None,
    ) -> None:
        if minimum_candidate_minus_baseline_endpoint_effect_lower is None:
            endpoint_threshold = (
                -0.02
                if minimum_safety_effect_lower is None
                else minimum_safety_effect_lower
            )
        elif minimum_safety_effect_lower is None:
            endpoint_threshold = minimum_candidate_minus_baseline_endpoint_effect_lower
        elif _same_numeric_value(
            minimum_candidate_minus_baseline_endpoint_effect_lower,
            minimum_safety_effect_lower,
        ):
            endpoint_threshold = minimum_candidate_minus_baseline_endpoint_effect_lower
        else:
            raise ValueError(
                "canonical and legacy endpoint-effect thresholds disagree"
            )

        object.__setattr__(
            self, "minimum_independent_campaigns", minimum_independent_campaigns
        )
        object.__setattr__(self, "minimum_feature_coverage", minimum_feature_coverage)
        object.__setattr__(self, "maximum_ood_fraction", maximum_ood_fraction)
        object.__setattr__(
            self, "maximum_calibration_gap_upper", maximum_calibration_gap_upper
        )
        object.__setattr__(
            self,
            "minimum_candidate_minus_baseline_endpoint_effect_lower",
            endpoint_threshold,
        )
        object.__setattr__(
            self, "maximum_control_loop_p95_ms", maximum_control_loop_p95_ms
        )

    @property
    def minimum_safety_effect_lower(self) -> float:
        """Deprecated alias for archived callers."""

        return self.minimum_candidate_minus_baseline_endpoint_effect_lower


@dataclass(frozen=True, init=False)
class GateEvidence:
    """Summary evidence supplied to the frozen comparison logic.

    ``evidence_basis`` separates a supplied gate-logic fixture from campaign-
    derived statistical bounds.  The gate does not estimate any of these
    summaries.  For campaign-derived bounds, the confidence level and declared
    multiple-testing treatment are required metadata.
    """

    schema_passed: object
    independent_campaigns: object
    calibration_gap_upper: object
    candidate_minus_baseline_endpoint_effect_lower: object
    control_loop_p95_ms: object
    endpoint_name: str
    evidence_basis: str
    confidence_level: object
    multiple_testing_adjustment: str | None

    def __init__(
        self,
        schema_passed: object,
        independent_campaigns: object,
        calibration_gap_upper: object = None,
        candidate_minus_baseline_endpoint_effect_lower: object = None,
        control_loop_p95_ms: object = None,
        *,
        safety_effect_lower: object = None,
        endpoint_name: str = "task_completion_rate",
        evidence_basis: str = GATE_LOGIC_FIXTURE,
        confidence_level: object = None,
        multiple_testing_adjustment: str | None = None,
    ) -> None:
        if candidate_minus_baseline_endpoint_effect_lower is None:
            endpoint_effect = safety_effect_lower
        elif safety_effect_lower is None:
            endpoint_effect = candidate_minus_baseline_endpoint_effect_lower
        elif _same_numeric_value(
            candidate_minus_baseline_endpoint_effect_lower,
            safety_effect_lower,
        ):
            endpoint_effect = candidate_minus_baseline_endpoint_effect_lower
        else:
            raise ValueError("canonical and legacy endpoint-effect evidence disagree")

        object.__setattr__(self, "schema_passed", schema_passed)
        object.__setattr__(self, "independent_campaigns", independent_campaigns)
        object.__setattr__(self, "calibration_gap_upper", calibration_gap_upper)
        object.__setattr__(
            self,
            "candidate_minus_baseline_endpoint_effect_lower",
            endpoint_effect,
        )
        object.__setattr__(self, "control_loop_p95_ms", control_loop_p95_ms)
        object.__setattr__(self, "endpoint_name", str(endpoint_name))
        object.__setattr__(self, "evidence_basis", str(evidence_basis))
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(
            self, "multiple_testing_adjustment", multiple_testing_adjustment
        )

    @property
    def safety_effect_lower(self) -> object:
        """Deprecated alias for archived callers."""

        return self.candidate_minus_baseline_endpoint_effect_lower


@dataclass(frozen=True)
class GateDecision:
    release_candidate: bool
    reasons: tuple[str, ...]
    feature_coverage: float
    ood_fraction: float
    delayed_shift_alarm: bool
    evidence_basis: str = "unavailable"
    endpoint_name: str = "unavailable"
    statistical_evidence_generated: bool = False
    statistical_authorization_supported: bool = False


class FrozenMahalanobisDetector:
    """Development-fitted multivariate guard with a frozen distance quantile."""

    def __init__(self, feature_columns: tuple[str, ...], quantile: float = 0.99):
        if not feature_columns:
            raise ValueError("at least one feature is required")
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1")
        self.feature_columns = tuple(feature_columns)
        self.quantile = float(quantile)

    def fit(self, frame: pd.DataFrame) -> "FrozenMahalanobisDetector":
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"development data are missing features: {sorted(missing)}")
        numeric = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        self.medians_ = numeric.median(axis=0)
        if self.medians_.isna().any():
            absent = self.medians_.index[self.medians_.isna()].tolist()
            raise ValueError(f"features contain no finite development values: {absent}")
        filled = numeric.fillna(self.medians_).to_numpy(float)
        self.center_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0, ddof=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardized = (filled - self.center_) / self.scale_
        covariance = LedoitWolf().fit(standardized)
        self.precision_ = covariance.precision_
        distances = self._distance_array(standardized)
        self.threshold_ = float(np.quantile(distances, self.quantile, method="higher"))
        self.fitted_rows_ = int(len(frame))
        return self

    def _require_fit(self) -> None:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit the detector before scoring")

    def _distance_array(self, standardized: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.maximum(
                np.einsum("ij,jk,ik->i", standardized, self.precision_, standardized),
                0.0,
            )
        )

    def feature_coverage(self, frame: pd.DataFrame) -> float:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing or not len(frame):
            return 0.0
        numeric = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        return float(np.isfinite(numeric.to_numpy(float)).mean())

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fit()
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"scoring data are missing features: {sorted(missing)}")
        numeric = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        filled = numeric.fillna(self.medians_).to_numpy(float)
        standardized = (filled - self.center_) / self.scale_
        return self._distance_array(standardized)

    def outlier_mask(self, frame: pd.DataFrame) -> np.ndarray:
        return self.score(frame) > self.threshold_


class CalibrationEProcess:
    """Anytime-valid one-sided monitor for a bounded calibration loss.

    For losses in [0, 1], Hoeffding's lemma makes
    ``exp(lambda*(loss-null_mean)-lambda**2/8)`` a valid e-increment under the
    null that conditional expected loss is at most ``null_mean``.  A fixed
    lambda is chosen before monitoring.  Crossing ``1/alpha`` triggers fallback.
    """

    def __init__(self, null_mean: float, *, alpha: float = 0.01, betting_fraction: float = 1.0):
        if not 0.0 <= null_mean <= 1.0:
            raise ValueError("null_mean must lie in [0, 1]")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if betting_fraction <= 0.0:
            raise ValueError("betting_fraction must be positive")
        self.null_mean = float(null_mean)
        self.alpha = float(alpha)
        self.betting_fraction = float(betting_fraction)
        self.reset()

    def reset(self) -> None:
        self.log_e_value_ = 0.0
        self.observations_ = 0
        self.alarm_ = False

    @property
    def e_value(self) -> float:
        return float(np.exp(min(self.log_e_value_, 700.0)))

    @property
    def alarm(self) -> bool:
        return bool(self.alarm_)

    def update(self, loss: float) -> float:
        loss = float(loss)
        if not 0.0 <= loss <= 1.0:
            raise ValueError("calibration loss must lie in [0, 1]")
        lam = self.betting_fraction
        self.log_e_value_ += lam * (loss - self.null_mean) - lam * lam / 8.0
        self.observations_ += 1
        if self.log_e_value_ >= np.log(1.0 / self.alpha):
            self.alarm_ = True
        return self.e_value


class EvidenceGate:
    """Fail-closed comparison of frozen summaries and shift diagnostics.

    A ``release_candidate`` result means that the supplied fields passed this
    comparison.  ``statistical_authorization_supported`` is deliberately more
    restrictive: it is true only for labelled campaign-derived bounds with the
    required inferential metadata.  Neither flag constructs statistical bounds.
    """

    def __init__(self, detector: FrozenMahalanobisDetector, criteria: GateCriteria):
        self.detector = detector
        self.criteria = criteria

    def evaluate(
        self,
        current_frame: pd.DataFrame,
        evidence: GateEvidence,
        *,
        delayed_shift_alarm: bool = False,
    ) -> GateDecision:
        reasons: list[str] = []
        try:
            coverage = self.detector.feature_coverage(current_frame)
            outlier = self.detector.outlier_mask(current_frame)
            ood_fraction = float(outlier.mean()) if len(outlier) else 1.0
        except (ValueError, RuntimeError):
            coverage = 0.0
            ood_fraction = 1.0
            reasons.append("domain_detector_unavailable")

        schema_available = isinstance(evidence.schema_passed, (bool, np.bool_))
        if not schema_available:
            reasons.append("schema_or_clock_audit_unavailable")
        elif not bool(evidence.schema_passed):
            reasons.append("schema_or_clock_audit_failed")

        campaigns_valid, campaigns = _finite_numeric(evidence.independent_campaigns)
        campaigns_valid = bool(
            campaigns_valid and campaigns >= 0.0 and campaigns.is_integer()
        )
        minimum_campaigns_valid, minimum_campaigns = _finite_numeric(
            self.criteria.minimum_independent_campaigns
        )
        minimum_campaigns_valid = bool(
            minimum_campaigns_valid
            and minimum_campaigns >= 0.0
            and minimum_campaigns.is_integer()
        )
        if not campaigns_valid:
            reasons.append("independent_campaign_count_unavailable")
        elif not minimum_campaigns_valid:
            reasons.append("gate_criteria_invalid")
        elif campaigns < minimum_campaigns:
            reasons.append("insufficient_independent_campaigns")

        coverage_threshold_valid, coverage_threshold = _finite_numeric(
            self.criteria.minimum_feature_coverage
        )
        if not coverage_threshold_valid or not 0.0 <= coverage_threshold <= 1.0:
            reasons.append("gate_criteria_invalid")
        elif coverage < coverage_threshold:
            reasons.append("insufficient_feature_coverage")

        ood_threshold_valid, ood_threshold = _finite_numeric(
            self.criteria.maximum_ood_fraction
        )
        if not ood_threshold_valid or not 0.0 <= ood_threshold <= 1.0:
            reasons.append("gate_criteria_invalid")
        elif ood_fraction > ood_threshold:
            reasons.append("out_of_domain_fraction_exceeded")

        calibration_valid, calibration_upper = _finite_numeric(
            evidence.calibration_gap_upper
        )
        calibration_threshold_valid, calibration_threshold = _finite_numeric(
            self.criteria.maximum_calibration_gap_upper
        )
        if not calibration_valid or not 0.0 <= calibration_upper <= 1.0:
            reasons.append("calibration_evidence_unavailable")
        elif not calibration_threshold_valid or not 0.0 <= calibration_threshold <= 1.0:
            reasons.append("gate_criteria_invalid")
        elif calibration_upper > calibration_threshold:
            reasons.append("calibration_bound_failed")

        endpoint_effect_valid, endpoint_effect_lower = _finite_numeric(
            evidence.candidate_minus_baseline_endpoint_effect_lower
        )
        endpoint_threshold_valid, endpoint_threshold = _finite_numeric(
            self.criteria.minimum_candidate_minus_baseline_endpoint_effect_lower
        )
        if not endpoint_effect_valid or not -1.0 <= endpoint_effect_lower <= 1.0:
            reasons.append("candidate_minus_baseline_endpoint_effect_unavailable")
        elif not endpoint_threshold_valid or not -1.0 <= endpoint_threshold <= 1.0:
            reasons.append("gate_criteria_invalid")
        elif endpoint_effect_lower < endpoint_threshold:
            reasons.append("candidate_minus_baseline_endpoint_noninferiority_failed")

        timing_valid, timing_p95_ms = _finite_numeric(evidence.control_loop_p95_ms)
        timing_threshold = self.criteria.maximum_control_loop_p95_ms
        timing_threshold_valid = bool(
            isinstance(timing_threshold, Real)
            and not isinstance(timing_threshold, (bool, np.bool_))
            and not np.isnan(float(timing_threshold))
            and float(timing_threshold) >= 0.0
        )
        if not timing_valid or timing_p95_ms < 0.0:
            reasons.append("control_loop_timing_evidence_unavailable")
        elif not timing_threshold_valid:
            reasons.append("gate_criteria_invalid")
        elif timing_p95_ms > float(timing_threshold):
            reasons.append("control_loop_deadline_failed")

        endpoint_name = str(evidence.endpoint_name).strip()
        if not endpoint_name or endpoint_name.lower() in {"none", "unavailable"}:
            reasons.append("endpoint_name_unavailable")

        evidence_basis = str(evidence.evidence_basis).strip()
        if evidence_basis not in _RECOGNIZED_EVIDENCE_BASES:
            reasons.append("evidence_basis_unrecognized")
        statistical_evidence_generated = bool(
            evidence_basis == CAMPAIGN_DERIVED_STATISTICAL_BOUNDS
        )
        if statistical_evidence_generated:
            confidence_valid, confidence_level = _finite_numeric(
                evidence.confidence_level
            )
            if not confidence_valid or not 0.0 < confidence_level < 1.0:
                reasons.append("statistical_confidence_level_unavailable")
            adjustment = evidence.multiple_testing_adjustment
            if not isinstance(adjustment, str) or not adjustment.strip():
                reasons.append("multiple_testing_adjustment_unavailable")

        delayed_shift_status_available = isinstance(
            delayed_shift_alarm, (bool, np.bool_)
        )
        delayed_shift_alarm_value = (
            bool(delayed_shift_alarm) if delayed_shift_status_available else False
        )
        if not delayed_shift_status_available:
            reasons.append("delayed_shift_status_unavailable")
        elif delayed_shift_alarm_value:
            reasons.append("delayed_outcome_shift_alarm")

        # Several checks can independently expose malformed criteria.  One
        # reason is sufficient and keeps machine-readable outputs stable.
        reasons = list(dict.fromkeys(reasons))
        release_candidate = not reasons

        return GateDecision(
            release_candidate=release_candidate,
            reasons=tuple(reasons),
            feature_coverage=float(coverage),
            ood_fraction=float(ood_fraction),
            delayed_shift_alarm=delayed_shift_alarm_value,
            evidence_basis=evidence_basis,
            endpoint_name=endpoint_name,
            statistical_evidence_generated=statistical_evidence_generated,
            statistical_authorization_supported=bool(
                release_candidate and statistical_evidence_generated
            ),
        )


def _clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact binomial interval for independent campaign-level outcomes."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must define a valid binomial count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if trials == 0:
        return np.nan, np.nan
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - tail, successes + 1, trials - successes)
    )
    return lower, upper


def evaluate_gate_campaigns(
    gate: EvidenceGate,
    streams: Mapping[str, pd.DataFrame],
    expected_shift: Mapping[str, bool],
    evidence: GateEvidence,
    *,
    batch_size: int = 50,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a frozen immediate guard at the independent-campaign level.

    Each campaign is processed in chronological, non-overlapping batches.  A
    campaign is counted as detected when any batch causes fail-closed fallback.
    The returned binomial intervals are meaningful only when the supplied
    campaigns are genuinely independent; generated stress replicates must be
    labelled simulation rather than field validation by the caller.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stream_ids = set(streams)
    label_ids = set(expected_shift)
    if stream_ids != label_ids:
        raise ValueError(
            "streams and expected_shift must have identical campaign identifiers"
        )
    if not stream_ids:
        raise ValueError("at least one campaign stream is required")

    rows: list[dict] = []
    for campaign_id in sorted(stream_ids):
        frame = streams[campaign_id].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"campaign {campaign_id!r} is empty")
        batch_rows: list[dict] = []
        for batch_index, start in enumerate(range(0, len(frame), batch_size)):
            stop = min(start + batch_size, len(frame))
            decision = gate.evaluate(frame.iloc[start:stop], evidence)
            batch_rows.append(
                {
                    "campaign_id": str(campaign_id),
                    "expected_shift": bool(expected_shift[campaign_id]),
                    "batch_index": int(batch_index),
                    "start_row": int(start),
                    "stop_row": int(stop),
                    "released": bool(decision.release_candidate),
                    "blocked": not bool(decision.release_candidate),
                    "reasons": "|".join(decision.reasons),
                    "feature_coverage": float(decision.feature_coverage),
                    "ood_fraction": float(decision.ood_fraction),
                    "endpoint_name": decision.endpoint_name,
                    "gate_evidence_basis": decision.evidence_basis,
                    "statistical_evidence_generated_upstream": bool(
                        decision.statistical_evidence_generated
                    ),
                    "statistical_authorization_supported": bool(
                        decision.statistical_authorization_supported
                    ),
                    "statistical_bounds_generated_by_this_function": False,
                }
            )
        campaign = pd.DataFrame(batch_rows)
        blocked_indices = campaign.loc[campaign["blocked"], "batch_index"]
        first_block = int(blocked_indices.iloc[0]) if len(blocked_indices) else np.nan
        campaign["campaign_blocked"] = bool(len(blocked_indices))
        campaign["first_block_batch"] = first_block
        campaign["detection_delay_rows"] = (
            int(first_block * batch_size) if np.isfinite(first_block) else np.nan
        )
        rows.extend(campaign.to_dict("records"))

    detail = pd.DataFrame(rows)
    campaigns = detail.groupby("campaign_id", as_index=False).first()
    stable = campaigns[~campaigns["expected_shift"]]
    shifted = campaigns[campaigns["expected_shift"]]
    false_alarms = int(stable["campaign_blocked"].sum())
    detections = int(shifted["campaign_blocked"].sum())
    false_alarm_ci = _clopper_pearson_interval(
        false_alarms, len(stable), confidence=confidence
    )
    detection_ci = _clopper_pearson_interval(
        detections, len(shifted), confidence=confidence
    )
    detected_delays = shifted.loc[
        shifted["campaign_blocked"], "detection_delay_rows"
    ].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "stable_campaigns": int(len(stable)),
                "shift_campaigns": int(len(shifted)),
                "false_alarm_campaigns": false_alarms,
                "detected_shift_campaigns": detections,
                "false_alarm_rate": float(false_alarms / len(stable))
                if len(stable)
                else np.nan,
                "false_alarm_rate_lower": false_alarm_ci[0],
                "false_alarm_rate_upper": false_alarm_ci[1],
                "detection_rate": float(detections / len(shifted))
                if len(shifted)
                else np.nan,
                "detection_rate_lower": detection_ci[0],
                "detection_rate_upper": detection_ci[1],
                "median_detection_delay_rows": float(np.median(detected_delays))
                if len(detected_delays)
                else np.nan,
                "confidence": float(confidence),
                "inference_unit": "campaign",
                "endpoint_name": str(evidence.endpoint_name),
                "gate_evidence_basis": str(evidence.evidence_basis),
                "statistical_evidence_generated_upstream": bool(
                    evidence.evidence_basis == CAMPAIGN_DERIVED_STATISTICAL_BOUNDS
                ),
                "statistical_bounds_generated_by_this_function": False,
                "validation_level": "gate_logic_operating_characteristics",
                "scope": (
                    "Exact binomial intervals assume independent supplied campaigns; "
                    "synthetic stress replicates are operating-characteristic evidence only. "
                    "This function tests gate logic and does not construct calibration, "
                    "endpoint-effect, or timing bounds."
                ),
            }
        ]
    )
    return detail, summary


def evaluate_calibration_monitor_campaigns(
    losses: Mapping[str, np.ndarray | list[float]],
    expected_shift: Mapping[str, bool],
    *,
    null_mean: float,
    alpha: float = 0.01,
    betting_fraction: float = 1.0,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fresh, frozen delayed-outcome monitors per campaign."""

    if set(losses) != set(expected_shift):
        raise ValueError(
            "losses and expected_shift must have identical campaign identifiers"
        )
    if not losses:
        raise ValueError("at least one loss stream is required")
    rows: list[dict] = []
    for campaign_id in sorted(losses):
        stream = np.asarray(losses[campaign_id], dtype=float)
        if stream.ndim != 1 or len(stream) == 0:
            raise ValueError(f"campaign {campaign_id!r} must have a nonempty 1-D stream")
        monitor = CalibrationEProcess(
            null_mean,
            alpha=alpha,
            betting_fraction=betting_fraction,
        )
        first_alarm = np.nan
        for index, value in enumerate(stream):
            e_value = monitor.update(float(value))
            if monitor.alarm and not np.isfinite(first_alarm):
                first_alarm = int(index)
            rows.append(
                {
                    "campaign_id": str(campaign_id),
                    "expected_shift": bool(expected_shift[campaign_id]),
                    "outcome_index": int(index),
                    "loss": float(value),
                    "e_value": float(e_value),
                    "alarm": bool(monitor.alarm),
                    "first_alarm_index": first_alarm,
                }
            )
    detail = pd.DataFrame(rows)
    campaigns = detail.groupby("campaign_id", as_index=False).last()
    stable = campaigns[~campaigns["expected_shift"]]
    shifted = campaigns[campaigns["expected_shift"]]
    false_alarms = int(stable["alarm"].sum())
    detections = int(shifted["alarm"].sum())
    false_alarm_ci = _clopper_pearson_interval(
        false_alarms, len(stable), confidence=confidence
    )
    detection_ci = _clopper_pearson_interval(
        detections, len(shifted), confidence=confidence
    )
    delays = shifted.loc[shifted["alarm"], "first_alarm_index"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "stable_campaigns": int(len(stable)),
                "shift_campaigns": int(len(shifted)),
                "false_alarm_campaigns": false_alarms,
                "detected_shift_campaigns": detections,
                "false_alarm_rate": float(false_alarms / len(stable))
                if len(stable)
                else np.nan,
                "false_alarm_rate_lower": false_alarm_ci[0],
                "false_alarm_rate_upper": false_alarm_ci[1],
                "detection_rate": float(detections / len(shifted))
                if len(shifted)
                else np.nan,
                "detection_rate_lower": detection_ci[0],
                "detection_rate_upper": detection_ci[1],
                "median_detection_delay_outcomes": float(np.median(delays))
                if len(delays)
                else np.nan,
                "confidence": float(confidence),
                "inference_unit": "campaign",
                "scope": (
                    "Each campaign starts a fresh prespecified e-process; campaign-rate "
                    "intervals require independent campaigns."
                ),
            }
        ]
    )
    return detail, summary
