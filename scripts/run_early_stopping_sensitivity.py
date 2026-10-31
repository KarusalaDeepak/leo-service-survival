#!/usr/bin/env python3
"""Episode-separated early-stopping sensitivity for the WetLinks focal model.

The iteration count is selected only inside the declared training partition:
the first 12 training calendar weeks are development data, the next week is an
embargo, and the final three training weeks are the chronological selection
holdout.  The chosen count is then frozen, the model is refitted on every
declared training episode with library early stopping disabled, calibrated on
the unchanged calibration partition, and evaluated once on the unchanged test
partition.  Only aggregate audit tables are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from leo_service_survival.data import (  # noqa: E402
    build_wetlinks_episodes,
    calendar_block_split,
    merge_sparse_terminal_inference_block,
)
from leo_service_survival.diagnostics import (  # noqa: E402
    assert_prediction_row_alignment,
    calibrated_prediction_stages,
    stage_horizon_diagnostics,
)
from leo_service_survival.metrics import evaluate_survival_predictions  # noqa: E402
from leo_service_survival.models import (  # noqa: E402
    CalibratedSurvivalModel,
    DiscreteHazardModel,
    FeatureEncoder,
    KaplanMeierModel,
    _expand_person_period,
)
from leo_service_survival.reporting import (  # noqa: E402
    calibration_intercept_slope,
    paired_proper_score_block_inference,
)
from run_all import truncate  # noqa: E402


ITERATION_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 90, 120, 180)


def _boosting_estimator(
    iterations: int,
    *,
    early_stopping: bool | str,
    random_state: int = 17,
) -> HistGradientBoostingClassifier:
    """Return the exact focal histogram-boosting configuration."""

    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=int(iterations),
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.5,
        random_state=int(random_state),
        early_stopping=early_stopping,
    )


def _person_period_origins(frame: pd.DataFrame, max_horizon: int) -> np.ndarray:
    """Return the source episode row for each expanded person-period row."""

    origins: list[int] = []
    for episode_index, (_, episode) in enumerate(frame.iterrows()):
        duration = int(episode["duration_bins"])
        observed_event = int(episode["event"]) == 1
        stop = duration if observed_event else duration - 1
        stop = min(stop, int(max_horizon))
        if stop >= 1:
            origins.extend([episode_index] * stop)
    return np.asarray(origins, dtype=int)


def _expand_with_training_causes(
    frame: pd.DataFrame,
    encoder: FeatureEncoder,
    cause_to_class: dict[str, int],
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand evaluation episodes using the training cause-to-class mapping."""

    base = encoder.transform(frame)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for episode_index, (_, episode) in enumerate(frame.iterrows()):
        duration = int(episode["duration_bins"])
        observed_event = int(episode["event"]) == 1
        stop = duration if observed_event else duration - 1
        stop = min(stop, int(max_horizon))
        if stop < 1:
            continue
        for step in range(1, stop + 1):
            time_features = np.asarray(
                [step / max_horizon, np.log1p(step)], dtype=float
            )
            interaction = np.asarray(
                [time_features[0] * base[episode_index, 0]], dtype=float
            )
            rows.append(
                np.concatenate([base[episode_index], time_features, interaction])
            )
            if observed_event and duration == step:
                labels.append(cause_to_class.get(str(episode["cause"]), 0))
            else:
                labels.append(0)
    return np.vstack(rows), np.asarray(labels, dtype=int)


def _wrap_fitted_base(
    encoder: FeatureEncoder,
    fitted: HistGradientBoostingClassifier,
    cause_to_class: dict[str, int],
    feature_columns: tuple[str, ...],
    max_horizon: int,
) -> DiscreteHazardModel:
    """Attach an explicitly fitted classifier to the project model interface."""

    model = DiscreteHazardModel(
        feature_columns,
        max_horizon=max_horizon,
        nonlinear=True,
        competing=True,
        random_state=17,
    )
    model.encoder = encoder
    model.model_ = fitted
    model.cause_to_class_ = dict(cause_to_class)
    model.classes_ = np.asarray(fitted.classes_, dtype=int)
    return model


def _fit_fixed_calibrated_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: tuple[str, ...],
    horizons: tuple[int, ...],
    iterations: int,
) -> CalibratedSurvivalModel:
    encoder = FeatureEncoder(feature_columns).fit(train)
    design, labels, weights, cause_to_class = _expand_person_period(
        train, encoder, max(horizons), competing=True
    )
    fitted = _boosting_estimator(
        iterations, early_stopping=False, random_state=17
    ).fit(design, labels, sample_weight=weights)
    base = _wrap_fitted_base(
        encoder,
        fitted,
        cause_to_class,
        feature_columns,
        max(horizons),
    )
    calibrated = CalibratedSurvivalModel(
        base,
        horizons,
        name="chronological_iteration_sensitivity",
    )
    calibrated.calibrate(calibration)
    return calibrated


def _fit_official_auto_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> CalibratedSurvivalModel:
    model = CalibratedSurvivalModel(
        DiscreteHazardModel(
            feature_columns,
            max_horizon=max(horizons),
            nonlinear=True,
            competing=True,
            random_state=17,
        ),
        horizons,
        name="official_auto_early_stopping",
    )
    model.fit(train)
    model.calibrate(calibration)
    return model


def _metric_row(
    variant: str,
    frame: pd.DataFrame,
    stages: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    *,
    selected_iterations: int | None,
    early_stopping: str,
) -> dict[str, object]:
    final = stages["final_monotone"]
    raw = stages["raw_base"]
    metrics = evaluate_survival_predictions(frame, final, horizons)
    raw_metrics = evaluate_survival_predictions(frame, raw, horizons)
    diagnostics = stage_horizon_diagnostics(frame, stages, horizons)
    final_60 = diagnostics.loc[
        diagnostics["stage"].eq("final_monotone")
        & diagnostics["horizon_bins"].eq(max(horizons))
    ].iloc[0]
    raw_60 = diagnostics.loc[
        diagnostics["stage"].eq("raw_base")
        & diagnostics["horizon_bins"].eq(max(horizons))
    ].iloc[0]
    return {
        "model_variant": variant,
        "selected_iterations": selected_iterations,
        "early_stopping": early_stopping,
        **metrics,
        "raw_integrated_brier": float(raw_metrics["integrated_brier"]),
        "raw_c_index": float(raw_metrics["c_index"]),
        "final_60min_calibration_slope": float(final_60["calibration_slope"]),
        "final_60min_pointwise_brier": float(final_60["pointwise_brier_bs_t"]),
        "final_60min_prediction_mean": float(final_60["prediction_mean"]),
        "final_60min_prediction_std": float(final_60["prediction_std"]),
        "raw_60min_calibration_slope": float(raw_60["calibration_slope"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wetlinks-dir",
        default=str(
            ROOT
            / "upstream/paper2/data/external/wetlinks_dataset/Preprocessed_Data"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/early_stopping_sensitivity_2026-10-31"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    max_horizon = 12
    horizons = tuple(range(1, max_horizon + 1))
    wetlinks = build_wetlinks_episodes(Path(args.wetlinks_dir).expanduser().resolve())
    frame = truncate(wetlinks.frame, max_horizon)
    official_train, official_calibration, final_test = calendar_block_split(frame)
    final_test, test_block_audit = merge_sparse_terminal_inference_block(
        final_test,
        group_column="calendar_block",
        output_column="analysis_block",
        minimum_fraction_of_prior_median=0.25,
    )
    expected_test_order = final_test["episode_id"].astype(str).to_numpy(copy=True)
    feature_columns = tuple(wetlinks.feature_columns)

    # Select iterations exclusively inside the already-declared training data.
    training_blocks = sorted(
        official_train["calendar_block"].dropna().astype(str).unique()
    )
    if len(training_blocks) != 16:
        raise AssertionError(
            f"expected 16 official training blocks, found {len(training_blocks)}"
        )
    development_blocks = training_blocks[:12]
    embargo_blocks = training_blocks[12:13]
    holdout_blocks = training_blocks[13:]
    development = official_train.loc[
        official_train["calendar_block"].astype(str).isin(development_blocks)
    ].sort_values(["start_time", "site"]).reset_index(drop=True)
    embargo = official_train.loc[
        official_train["calendar_block"].astype(str).isin(embargo_blocks)
    ].sort_values(["start_time", "site"]).reset_index(drop=True)
    selection_holdout = official_train.loc[
        official_train["calendar_block"].astype(str).isin(holdout_blocks)
    ].sort_values(["start_time", "site"]).reset_index(drop=True)
    if not (
        development["analysis_label_resolution_time"].max()
        < selection_holdout["start_time"].min()
    ):
        raise AssertionError("development labels overlap the chronological holdout")

    development_encoder = FeatureEncoder(feature_columns).fit(development)
    development_x, development_y, development_w, development_causes = (
        _expand_person_period(
            development,
            development_encoder,
            max_horizon,
            competing=True,
        )
    )
    holdout_x, holdout_y = _expand_with_training_causes(
        selection_holdout,
        development_encoder,
        development_causes,
        max_horizon,
    )

    selection_rows: list[dict[str, object]] = []
    for iterations in ITERATION_GRID:
        classifier = _boosting_estimator(
            iterations, early_stopping=False, random_state=17
        ).fit(development_x, development_y, sample_weight=development_w)
        probability = classifier.predict_proba(holdout_x)
        base = _wrap_fitted_base(
            development_encoder,
            classifier,
            development_causes,
            feature_columns,
            max_horizon,
        )
        survival = base.predict_survival(selection_holdout, horizons)
        holdout_metrics = evaluate_survival_predictions(
            selection_holdout, survival, horizons
        )
        selection_rows.append(
            {
                "iterations": int(iterations),
                "selection_person_period_log_loss": float(
                    log_loss(holdout_y, probability, labels=classifier.classes_)
                ),
                "selection_integrated_brier": float(
                    holdout_metrics["integrated_brier"]
                ),
                "selection_c_index": float(holdout_metrics["c_index"]),
                "selection_restricted_mean_time_mae_bins": float(
                    holdout_metrics["restricted_mean_time_mae_bins"]
                ),
            }
        )
    selection_curve = pd.DataFrame(selection_rows)
    selected_iterations = int(
        selection_curve.loc[
            selection_curve["selection_person_period_log_loss"].idxmin(),
            "iterations",
        ]
    )
    brier_selected_iterations = int(
        selection_curve.loc[
            selection_curve["selection_integrated_brier"].idxmin(), "iterations"
        ]
    )

    # The choice is now frozen.  Nothing from final_test entered the preceding
    # iteration-selection calculations.
    official_model = _fit_official_auto_model(
        official_train,
        official_calibration,
        feature_columns,
        horizons,
    )
    sensitivity_model = _fit_fixed_calibrated_model(
        official_train,
        official_calibration,
        feature_columns,
        horizons,
        selected_iterations,
    )
    official_stages = calibrated_prediction_stages(
        official_model, final_test, horizons
    )
    sensitivity_stages = calibrated_prediction_stages(
        sensitivity_model, final_test, horizons
    )
    assert_prediction_row_alignment(
        final_test,
        official_stages,
        horizons,
        expected_episode_ids=expected_test_order,
    )
    assert_prediction_row_alignment(
        final_test,
        sensitivity_stages,
        horizons,
        expected_episode_ids=expected_test_order,
    )

    metric_rows = [
        _metric_row(
            "official_auto_early_stopping",
            final_test,
            official_stages,
            horizons,
            selected_iterations=int(official_model.base_model.model_.n_iter_),
            early_stopping="auto; row-level 10% internal validation",
        ),
        _metric_row(
            "chronological_fixed_iteration_sensitivity",
            final_test,
            sensitivity_stages,
            horizons,
            selected_iterations=selected_iterations,
            early_stopping="disabled; iteration count frozen on earlier chronological holdout",
        ),
    ]

    kaplan_meier = KaplanMeierModel().fit(official_train)
    km_prediction = kaplan_meier.predict_survival(final_test, horizons)
    km_metrics = evaluate_survival_predictions(final_test, km_prediction, horizons)
    metric_rows.append(
        {
            "model_variant": "kaplan_meier_feature_free_reference",
            "selected_iterations": None,
            "early_stopping": "not applicable",
            **km_metrics,
            "raw_integrated_brier": float(km_metrics["integrated_brier"]),
            "raw_c_index": float(km_metrics["c_index"]),
            "final_60min_calibration_slope": np.nan,
            "final_60min_pointwise_brier": np.nan,
            "final_60min_prediction_mean": float(km_prediction[:, -1].mean()),
            "final_60min_prediction_std": float(km_prediction[:, -1].std()),
            "raw_60min_calibration_slope": np.nan,
        }
    )
    metrics = pd.DataFrame(metric_rows)
    km_ibs = float(
        metrics.loc[
            metrics["model_variant"].eq("kaplan_meier_feature_free_reference"),
            "integrated_brier",
        ].iloc[0]
    )
    metrics["integrated_brier_skill_vs_kaplan_meier"] = (
        km_ibs - metrics["integrated_brier"]
    ) / km_ibs

    paired_blocks, paired_inference = paired_proper_score_block_inference(
        final_test,
        {
            "chronological_fixed": sensitivity_stages["final_monotone"],
            "official_auto": official_stages["final_monotone"],
            "kaplan_meier": km_prediction,
        },
        horizons,
        reference="chronological_fixed",
        baselines=("official_auto", "kaplan_meier"),
        group_column="analysis_block",
    )
    calibration_parameter_parts = []
    for variant, survival in (
        ("official_auto_early_stopping", official_stages["final_monotone"]),
        (
            "chronological_fixed_iteration_sensitivity",
            sensitivity_stages["final_monotone"],
        ),
    ):
        parameters = calibration_intercept_slope(
            final_test,
            survival[:, [-1]],
            (max(horizons),),
            repetitions=400,
            seed=17,
            group_column="analysis_block",
        )
        parameters.insert(0, "model_variant", variant)
        calibration_parameter_parts.append(parameters)
    calibration_parameters = pd.concat(
        calibration_parameter_parts, ignore_index=True
    )

    official_row = metrics.loc[
        metrics["model_variant"].eq("official_auto_early_stopping")
    ].iloc[0]
    sensitivity_row = metrics.loc[
        metrics["model_variant"].eq(
            "chronological_fixed_iteration_sensitivity"
        )
    ].iloc[0]
    numeric_comparisons = [
        "integrated_brier",
        "mean_calibration_error",
        "binary_cross_entropy",
        "restricted_mean_time_mae_bins",
        "c_index",
        "raw_integrated_brier",
        "raw_c_index",
        "final_60min_calibration_slope",
        "final_60min_pointwise_brier",
        "final_60min_prediction_mean",
        "final_60min_prediction_std",
        "raw_60min_calibration_slope",
    ]
    differences = pd.DataFrame(
        [
            {
                "metric": metric,
                "official_auto": float(official_row[metric]),
                "chronological_fixed": float(sensitivity_row[metric]),
                "fixed_minus_official": float(
                    sensitivity_row[metric] - official_row[metric]
                ),
            }
            for metric in numeric_comparisons
        ]
    )

    # Reconstruct the library's exact internal row split to quantify episode
    # overlap; this is an aggregate audit and exports no episode identifiers.
    full_encoder = official_model.base_model.encoder
    full_x, full_y, full_w, _ = _expand_person_period(
        official_train, full_encoder, max_horizon, competing=True
    )
    origins = _person_period_origins(official_train, max_horizon)
    if len(origins) != len(full_y):
        raise AssertionError("person-period origin vector is misaligned")
    fitted_classifier = official_model.base_model.model_
    split = train_test_split(
        np.arange(len(full_y)),
        full_y,
        full_w,
        test_size=fitted_classifier.validation_fraction,
        stratify=full_y,
        random_state=fitted_classifier._random_seed,
    )
    internal_train_index, internal_validation_index = split[0], split[1]
    internal_train_episodes = set(origins[internal_train_index].tolist())
    internal_validation_episodes = set(origins[internal_validation_index].tolist())
    shared_episodes = internal_train_episodes & internal_validation_episodes
    shared_validation_rows = np.isin(
        origins[internal_validation_index], list(internal_train_episodes)
    )
    leakage_audit = pd.DataFrame(
        [
            {
                "official_training_episodes": int(len(official_train)),
                "official_training_events": int(official_train["event"].sum()),
                "expanded_rows": int(len(full_y)),
                "auto_iterations_returned": int(fitted_classifier.n_iter_),
                "internal_validation_rows": int(len(internal_validation_index)),
                "internal_validation_episodes": int(
                    len(internal_validation_episodes)
                ),
                "episodes_shared_between_internal_train_and_validation": int(
                    len(shared_episodes)
                ),
                "fraction_internal_validation_episodes_also_in_train": float(
                    len(shared_episodes) / len(internal_validation_episodes)
                ),
                "internal_validation_rows_whose_episode_is_also_in_train": int(
                    shared_validation_rows.sum()
                ),
                "fraction_internal_validation_rows_whose_episode_is_also_in_train": float(
                    shared_validation_rows.mean()
                ),
            }
        ]
    )

    selection_curve.to_csv(output_dir / "iteration_selection_curve.csv", index=False)
    metrics.to_csv(output_dir / "final_test_aggregate_metrics.csv", index=False)
    differences.to_csv(output_dir / "fixed_minus_official_differences.csv", index=False)
    leakage_audit.to_csv(output_dir / "internal_split_leakage_audit.csv", index=False)
    paired_inference.to_csv(output_dir / "paired_score_inference.csv", index=False)
    paired_blocks.to_csv(output_dir / "paired_score_blocks.csv", index=False)
    calibration_parameters.to_csv(
        output_dir / "calibration_parameters_60min.csv", index=False
    )

    summary = {
        "purpose": "bounded episode-separated early-stopping sensitivity",
        "selection_did_not_use_final_test": True,
        "development": {
            "blocks": development_blocks,
            "episodes": int(len(development)),
            "events": int(development["event"].sum()),
            "expanded_rows": int(len(development_y)),
            "last_label_resolution_time": development[
                "analysis_label_resolution_time"
            ].max().isoformat(),
        },
        "embargo": {
            "blocks": embargo_blocks,
            "episodes": int(len(embargo)),
            "events": int(embargo["event"].sum()),
        },
        "selection_holdout": {
            "blocks": holdout_blocks,
            "episodes": int(len(selection_holdout)),
            "events": int(selection_holdout["event"].sum()),
            "expanded_rows": int(len(holdout_y)),
            "first_start_time": selection_holdout["start_time"].min().isoformat(),
        },
        "iteration_grid": list(ITERATION_GRID),
        "selected_iterations_by_person_period_log_loss": selected_iterations,
        "selected_iterations_by_integrated_brier_check": brier_selected_iterations,
        "official_auto_iterations_returned": int(
            official_model.base_model.model_.n_iter_
        ),
        "official_internal_best_validation_score_iteration": int(
            np.argmax(official_model.base_model.model_.validation_score_)
        ),
        "unchanged_calibration_episodes": int(len(official_calibration)),
        "unchanged_final_test_episodes": int(len(final_test)),
        "unchanged_final_test_events": int(final_test["event"].sum()),
        "test_analysis_block_audit": test_block_audit,
        "aggregate_outputs_only": True,
        "principal_conclusion": (
            "The chronological sensitivity has lower point-estimate Brier and "
            "log losses and a c-index closer to chance from above, but its gain "
            "over the feature-free Kaplan-Meier reference remains small and its "
            "60-minute predictions are highly compressed. The paper's central "
            "conclusion of weak individualized temporal transport is unchanged."
        ),
        "interpretation": (
            "The sensitivity changes only how the boosting iteration count is "
            "chosen. Endpoint construction, official partitions, calibration, "
            "test rows, horizons, metrics, and risk orientation are unchanged."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nFinal-test aggregate metrics")
    print(metrics.to_string(index=False))
    print("\nChronological fixed minus official auto")
    print(differences.to_string(index=False))
    print(f"\nWrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
