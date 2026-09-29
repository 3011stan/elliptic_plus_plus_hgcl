from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from hgcl.studies.s003.statistics import (
    CANONICAL_SEEDS,
    PRIMARY_COMPARISONS,
    AblationAttribution,
    ClaimGateResult,
    PairedDifference,
    apply_holm_bonferroni,
    compute_ablation_attribution,
    compute_paired_difference,
    compute_temporal_series,
    evaluate_claim_gate,
)


def test_paired_difference_with_five_complete_seeds() -> None:
    # Known scores for S003 and baseline
    scores_s003 = {11: 0.72, 23: 0.75, 37: 0.71, 53: 0.76, 71: 0.74}
    scores_base = {11: 0.68, 23: 0.70, 37: 0.69, 53: 0.71, 71: 0.70}

    res = compute_paired_difference(
        scores_s003,
        scores_base,
        metric="f1_illicit",
        method_a="s003_txgcl",
        method_b="gcpal",
        fraction=0.01,
    )

    assert res.is_complete is True
    assert res.seeds == CANONICAL_SEEDS
    expected_diffs = (0.04, 0.05, 0.02, 0.05, 0.04)
    for obs, exp in zip(res.differences, expected_diffs):
        assert pytest.approx(obs, abs=1e-5) == exp

    expected_mean = float(np.mean(expected_diffs))
    expected_std = float(np.std(expected_diffs, ddof=1))
    assert pytest.approx(res.mean_difference, abs=1e-5) == expected_mean
    assert pytest.approx(res.std_difference, abs=1e-5) == expected_std

    # Check against scipy.stats.ttest_rel
    scipy_stat, scipy_p = scipy.stats.ttest_rel(
        [scores_s003[s] for s in CANONICAL_SEEDS],
        [scores_base[s] for s in CANONICAL_SEEDS],
    )
    assert pytest.approx(res.t_statistic, abs=1e-5) == scipy_stat
    assert pytest.approx(res.p_value, abs=1e-5) == scipy_p

    # 95% CI with df=4
    se = expected_std / np.sqrt(5)
    t_crit = scipy.stats.t.ppf(0.975, df=4)
    assert pytest.approx(res.ci_95_lower, abs=1e-5) == expected_mean - t_crit * se
    assert pytest.approx(res.ci_95_upper, abs=1e-5) == expected_mean + t_crit * se

    # Cohen's d_z = mean / std
    assert pytest.approx(res.cohen_dz, abs=1e-5) == expected_mean / expected_std
    assert res.positive_seed_count == 5


def test_paired_difference_incomplete_behavior_fr044() -> None:
    # Only 4 seeds available (missing seed 71)
    scores_s003 = {11: 0.72, 23: 0.75, 37: 0.71, 53: 0.76}
    scores_base = {11: 0.68, 23: 0.70, 37: 0.69, 53: 0.71, 71: 0.70}

    res = compute_paired_difference(
        scores_s003,
        scores_base,
        metric="f1_illicit",
        method_a="s003_txgcl",
        method_b="gcpal",
        fraction=0.01,
    )

    # FR-044: Incomplete pairs must NOT calculate p-value, CI, effect size or allow superiority
    assert res.is_complete is False
    assert res.mean_difference is not None  # Descriptive value allowed
    assert res.std_difference is None
    assert res.p_value is None
    assert res.ci_95_lower is None
    assert res.ci_95_upper is None
    assert res.cohen_dz is None
    assert "incomplete pair" in res.reason.lower()


def test_paired_difference_degenerate_cases() -> None:
    # 1. Zero difference across all seeds
    scores_zero = {s: 0.70 for s in CANONICAL_SEEDS}
    res_zero = compute_paired_difference(
        scores_zero, scores_zero, metric="f1_illicit", method_b="base", fraction=0.01
    )
    assert res_zero.is_complete is True
    assert res_zero.mean_difference == 0.0
    assert res_zero.t_statistic == 0.0
    assert res_zero.p_value == 1.0
    assert res_zero.cohen_dz == 0.0

    # 2. Constant positive difference across all seeds
    scores_higher = {s: 0.75 for s in CANONICAL_SEEDS}
    res_const = compute_paired_difference(
        scores_higher, scores_zero, metric="f1_illicit", method_b="base", fraction=0.01
    )
    assert res_const.is_complete is True
    assert pytest.approx(res_const.mean_difference, abs=1e-5) == 0.05
    assert res_const.p_value == 0.0
    assert res_const.cohen_dz == float("inf")


def test_holm_bonferroni_step_down_adjustment() -> None:
    # 4 comparisons with predefined unadjusted p-values
    # Say p = [0.01, 0.04, 0.03, 0.10]
    # Sorted:
    # rank 0: 0.01 * 4 = 0.04
    # rank 1: 0.03 * 3 = 0.09
    # rank 2: 0.04 * 2 = 0.08 -> running max = 0.09
    # rank 3: 0.10 * 1 = 0.10 -> running max = 0.10
    c0 = PairedDifference("c0", "a", "b", 0.01, "f1", CANONICAL_SEEDS, (), (), (), True, p_value=0.01)
    c1 = PairedDifference("c1", "a", "b", 0.05, "f1", CANONICAL_SEEDS, (), (), (), True, p_value=0.04)
    c2 = PairedDifference("c2", "a", "c", 0.01, "f1", CANONICAL_SEEDS, (), (), (), True, p_value=0.03)
    c3 = PairedDifference("c3", "a", "c", 0.05, "f1", CANONICAL_SEEDS, (), (), (), True, p_value=0.10)

    adjusted = apply_holm_bonferroni([c0, c1, c2, c3], family_size=4)
    assert len(adjusted) == 4
    # Verify order is preserved
    assert adjusted[0].comparison_id == "c0"
    assert adjusted[1].comparison_id == "c1"
    assert adjusted[2].comparison_id == "c2"
    assert adjusted[3].comparison_id == "c3"

    assert pytest.approx(adjusted[0].p_value_holm, abs=1e-5) == 0.04
    assert pytest.approx(adjusted[1].p_value_holm, abs=1e-5) == 0.09
    assert pytest.approx(adjusted[2].p_value_holm, abs=1e-5) == 0.09
    assert pytest.approx(adjusted[3].p_value_holm, abs=1e-5) == 0.10


def test_claim_gate_strict_scientific_criteria_fr045() -> None:
    # Case 1: Valid superiority in F1 illicit with no degradation in MCC
    valid_f1 = PairedDifference(
        "f1", "s003_txgcl", "gcpal", 0.01, "f1_illicit", CANONICAL_SEEDS,
        scores_a=(0.7, 0.7, 0.7, 0.7, 0.7), scores_b=(0.6, 0.6, 0.6, 0.6, 0.6),
        differences=(0.1, 0.1, 0.1, 0.1, 0.1), is_complete=True,
        mean_difference=0.10, ci_95_lower=0.05, ci_95_upper=0.15,
        p_value=0.005, p_value_holm=0.02, positive_seed_count=5,
    )
    neutral_mcc = PairedDifference(
        "mcc", "s003_txgcl", "gcpal", 0.01, "mcc", CANONICAL_SEEDS,
        scores_a=(0.6, 0.6, 0.6, 0.6, 0.6), scores_b=(0.6, 0.6, 0.6, 0.6, 0.6),
        differences=(0.0, 0.0, 0.0, 0.0, 0.0), is_complete=True,
        mean_difference=0.0, ci_95_lower=-0.02, ci_95_upper=0.02,
        p_value=1.0, p_value_holm=1.0, positive_seed_count=0,
    )

    gate_res = evaluate_claim_gate(valid_f1, neutral_mcc)
    assert gate_res.is_f1_superior is True
    assert gate_res.is_superior is True
    assert gate_res.is_label_efficient is True  # 1% fraction

    # Case 2: Positive mean but 95% CI lower <= 0 -> Fails
    ci_cross_zero_f1 = PairedDifference(
        "f1", "s003_txgcl", "gcpal", 0.01, "f1_illicit", CANONICAL_SEEDS,
        scores_a=(), scores_b=(), differences=(), is_complete=True,
        mean_difference=0.05, ci_95_lower=-0.01, ci_95_upper=0.11,
        p_value=0.04, p_value_holm=0.04, positive_seed_count=4,
    )
    res_ci = evaluate_claim_gate(ci_cross_zero_f1, neutral_mcc)
    assert res_ci.is_f1_superior is False
    assert res_ci.is_superior is False

    # Case 3: Positive mean, CI > 0, but positive in only 3/5 seeds -> Fails
    few_pos_seeds_f1 = PairedDifference(
        "f1", "s003_txgcl", "gcpal", 0.01, "f1_illicit", CANONICAL_SEEDS,
        scores_a=(), scores_b=(), differences=(), is_complete=True,
        mean_difference=0.05, ci_95_lower=0.01, ci_95_upper=0.09,
        p_value=0.02, p_value_holm=0.04, positive_seed_count=3,
    )
    res_seeds = evaluate_claim_gate(few_pos_seeds_f1, neutral_mcc)
    assert res_seeds.is_f1_superior is False
    assert res_seeds.is_superior is False

    # Case 4: F1 passes, but MCC has statistically significant degradation -> Fails
    degraded_mcc = PairedDifference(
        "mcc", "s003_txgcl", "gcpal", 0.01, "mcc", CANONICAL_SEEDS,
        scores_a=(), scores_b=(), differences=(), is_complete=True,
        mean_difference=-0.08, ci_95_lower=-0.12, ci_95_upper=-0.04,
        p_value=0.005, p_value_holm=0.01, positive_seed_count=0,
    )
    res_deg = evaluate_claim_gate(valid_f1, degraded_mcc)
    assert res_deg.is_f1_superior is False
    assert res_deg.is_superior is False
    assert any("statistically significant degradation" in r for r in res_deg.reasons)

    # Case 5: Incomplete comparison cannot claim superiority (FR-044)
    incomplete_f1 = PairedDifference(
        "f1", "s003_txgcl", "gcpal", 0.01, "f1_illicit", CANONICAL_SEEDS[:4],
        scores_a=(), scores_b=(), differences=(), is_complete=False,
        mean_difference=0.10, positive_seed_count=4,
    )
    res_inc = evaluate_claim_gate(incomplete_f1, neutral_mcc)
    assert res_inc.is_superior is False


def test_ablation_attribution_computation() -> None:
    target_f1 = {11: 0.75, 23: 0.77, 37: 0.74, 53: 0.78, 71: 0.76}
    target_mcc = {11: 0.65, 23: 0.67, 37: 0.64, 53: 0.68, 71: 0.66}
    ref_f1 = {11: 0.70, 23: 0.71, 37: 0.69, 53: 0.72, 71: 0.70}
    ref_mcc = {11: 0.60, 23: 0.61, 37: 0.59, 53: 0.62, 71: 0.60}

    attr = compute_ablation_attribution(
        target_f1,
        target_mcc,
        ref_f1,
        ref_mcc,
        attribution_type="representation",
        target_variant="h_concat_x",
        reference_variant="h_only",
    )

    assert attr.attribution_type == "representation"
    assert attr.target_variant == "h_concat_x"
    assert attr.reference_variant == "h_only"
    assert len(attr.f1_differences) == 5
    assert pytest.approx(attr.f1_mean_difference, abs=1e-5) == 0.056
    assert attr.f1_relative_gain_percent > 0.0
    assert attr.mcc_relative_gain_percent > 0.0


def test_temporal_series_per_snapshot() -> None:
    preds = {
        35: ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]),
        36: ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]),  # No illicit nodes
    }
    series = compute_temporal_series(preds, threshold=0.5)
    assert 35 in series and 36 in series
    assert series[35]["f1_illicit"] == 1.0
    assert series[35]["mcc"] == 1.0
    # Step 36 has only licit nodes -> mcc is None with reason
    assert series[36]["mcc"] is None
    assert series[36]["reason"] == "metric requires both known classes"
