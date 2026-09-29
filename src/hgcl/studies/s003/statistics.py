"""Predeclared paired statistical analysis, Holm correction, ablation attributions, and claim gates for S003."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.stats

from .evaluation import binary_metrics


PRIMARY_COMPARISONS = (
    ("s003_txgcl", "gcpal", 0.01),
    ("s003_txgcl", "gcpal", 0.05),
    ("s003_txgcl", "inspection_l_dgi", 0.01),
    ("s003_txgcl", "inspection_l_dgi", 0.05),
)

CANONICAL_SEEDS = (11, 23, 37, 53, 71)


class StatisticalContractError(ValueError):
    """Raised when statistical evaluation violates pre-declared protocols."""


@dataclass(frozen=True)
class PairedDifference:
    """Paired comparison between method A and method B across identical seeds."""

    comparison_id: str
    method_a: str
    method_b: str
    fraction: float
    metric: str
    seeds: tuple[int, ...]
    scores_a: tuple[float, ...]
    scores_b: tuple[float, ...]
    differences: tuple[float, ...]
    is_complete: bool
    mean_difference: float | None = None
    std_difference: float | None = None
    ci_95_lower: float | None = None
    ci_95_upper: float | None = None
    t_statistic: float | None = None
    p_value: float | None = None
    p_value_holm: float | None = None
    cohen_dz: float | None = None
    positive_seed_count: int = 0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_paired_difference(
    scores_a_by_seed: Mapping[int, float],
    scores_b_by_seed: Mapping[int, float],
    *,
    metric: str,
    method_a: str = "s003_txgcl",
    method_b: str,
    fraction: float,
    required_seeds: Sequence[int] = CANONICAL_SEEDS,
) -> PairedDifference:
    """Compute paired differences, bilateral t-test, 95% CI, and Cohen's d_z across seeds."""
    comparison_id = f"{method_a}_vs_{method_b}:{fraction}:{metric}"
    common_seeds = tuple(s for s in required_seeds if s in scores_a_by_seed and s in scores_b_by_seed)

    # Check completeness: all required seeds must be present with valid numeric values
    has_all_seeds = len(common_seeds) == len(required_seeds)
    all_numeric = has_all_seeds and all(
        scores_a_by_seed[s] is not None
        and scores_b_by_seed[s] is not None
        and not np.isnan(scores_a_by_seed[s])
        and not np.isnan(scores_b_by_seed[s])
        for s in common_seeds
    )

    if not all_numeric:
        # Incomplete pair: descriptive only, no p-value, CI, or effect size (FR-044)
        available_seeds = tuple(s for s in common_seeds if scores_a_by_seed[s] is not None and scores_b_by_seed[s] is not None)
        scores_a = tuple(float(scores_a_by_seed[s]) for s in available_seeds)
        scores_b = tuple(float(scores_b_by_seed[s]) for s in available_seeds)
        differences = tuple(a - b for a, b in zip(scores_a, scores_b))
        mean_diff = float(np.mean(differences)) if differences else None
        pos_count = sum(1 for d in differences if d > 0)
        return PairedDifference(
            comparison_id=comparison_id,
            method_a=method_a,
            method_b=method_b,
            fraction=fraction,
            metric=metric,
            seeds=available_seeds,
            scores_a=scores_a,
            scores_b=scores_b,
            differences=differences,
            is_complete=False,
            mean_difference=mean_diff,
            positive_seed_count=pos_count,
            reason=f"incomplete pair: requires all 5 seeds {list(required_seeds)}",
        )

    seeds = tuple(required_seeds)
    scores_a = tuple(float(scores_a_by_seed[s]) for s in seeds)
    scores_b = tuple(float(scores_b_by_seed[s]) for s in seeds)
    differences = tuple(a - b for a, b in zip(scores_a, scores_b))
    n = len(differences)

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    positive_count = sum(1 for d in differences if d > 0)

    if std_diff == 0.0:
        ci_lower = mean_diff
        ci_upper = mean_diff
        if mean_diff == 0.0:
            t_stat = 0.0
            p_val = 1.0
            cohen_dz = 0.0
        elif mean_diff > 0.0:
            t_stat = float("inf")
            p_val = 0.0
            cohen_dz = float("inf")
        else:
            t_stat = float("-inf")
            p_val = 0.0
            cohen_dz = float("-inf")
    else:
        se = std_diff / np.sqrt(n)
        df = n - 1
        t_crit = float(scipy.stats.t.ppf(0.975, df=df))
        ci_lower = float(mean_diff - t_crit * se)
        ci_upper = float(mean_diff + t_crit * se)
        t_stat = float(mean_diff / se)
        p_val = float(2.0 * (1.0 - scipy.stats.t.cdf(abs(t_stat), df=df)))
        cohen_dz = float(mean_diff / std_diff)

    return PairedDifference(
        comparison_id=comparison_id,
        method_a=method_a,
        method_b=method_b,
        fraction=fraction,
        metric=metric,
        seeds=seeds,
        scores_a=scores_a,
        scores_b=scores_b,
        differences=differences,
        is_complete=True,
        mean_difference=mean_diff,
        std_difference=std_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        t_statistic=t_stat,
        p_value=p_val,
        cohen_dz=cohen_dz,
        positive_seed_count=positive_count,
    )


def apply_holm_bonferroni(
    comparisons: Sequence[PairedDifference],
    *,
    family_size: int = 4,
) -> tuple[PairedDifference, ...]:
    """Apply step-down Holm-Bonferroni correction to a family of comparisons."""
    complete = [comp for comp in comparisons if comp.is_complete and comp.p_value is not None]
    incomplete = [comp for comp in comparisons if not comp.is_complete or comp.p_value is None]

    # If any comparison in the family is incomplete, mark them accordingly
    if len(complete) != len(comparisons):
        # Return as-is without Holm adjustment
        return tuple(comparisons)

    # Sort complete by unadjusted p-value ascending
    sorted_complete = sorted(complete, key=lambda c: (c.p_value if c.p_value is not None else 1.0))
    m = family_size

    adjusted_pairs: list[PairedDifference] = []
    running_max = 0.0
    for rank, comp in enumerate(sorted_complete):
        # multiplier: m - rank
        raw_p = comp.p_value if comp.p_value is not None else 1.0
        multiplier = m - rank
        adj_p = min(1.0, raw_p * multiplier)
        adj_p = max(adj_p, running_max)
        running_max = adj_p

        d = comp.to_dict()
        d["p_value_holm"] = float(adj_p)
        adjusted_pairs.append(PairedDifference(**d))

    # Restore original ordering
    order_map = {comp.comparison_id: comp for comp in adjusted_pairs}
    result = tuple(order_map[c.comparison_id] for c in comparisons)
    return result


@dataclass(frozen=True)
class ClaimGateResult:
    """Evaluation of scientific superiority and label efficiency claim gates."""

    comparison_id: str
    fraction: float
    is_f1_superior: bool
    is_mcc_superior: bool
    is_superior: bool
    is_label_efficient: bool
    f1_details: dict[str, Any]
    mcc_details: dict[str, Any]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_claim_gate(
    f1_comp: PairedDifference,
    mcc_comp: PairedDifference,
    *,
    alpha: float = 0.05,
) -> ClaimGateResult:
    """Evaluate whether S003-TxGCL meets strict statistical claim gates (FR-045)."""
    comparison_id = f"{f1_comp.method_a}_vs_{f1_comp.method_b}:{f1_comp.fraction}"
    fraction = f1_comp.fraction
    reasons: list[str] = []

    def _check_superiority(comp: PairedDifference) -> tuple[bool, dict[str, Any]]:
        details = {
            "is_complete": comp.is_complete,
            "mean_difference": comp.mean_difference,
            "ci_95_lower": comp.ci_95_lower,
            "p_value_holm": comp.p_value_holm,
            "positive_seed_count": comp.positive_seed_count,
        }
        if not comp.is_complete:
            return False, {**details, "fail_reason": "incomplete pairs"}
        if comp.mean_difference is None or comp.mean_difference <= 0.0:
            return False, {**details, "fail_reason": "mean difference is not positive"}
        if comp.ci_95_lower is None or comp.ci_95_lower <= 0.0:
            return False, {**details, "fail_reason": "95% CI lower bound is not strictly positive"}
        if comp.p_value_holm is None or comp.p_value_holm >= alpha:
            return False, {**details, "fail_reason": f"Holm-adjusted p-value {comp.p_value_holm} >= {alpha}"}
        if comp.positive_seed_count < 4:
            return False, {**details, "fail_reason": f"positive in only {comp.positive_seed_count}/5 seeds (requires >= 4)"}
        return True, {**details, "pass": True}

    def _check_significant_degradation(comp: PairedDifference) -> bool:
        if not comp.is_complete:
            return False
        if comp.mean_difference is not None and comp.mean_difference < 0.0:
            if comp.ci_95_upper is not None and comp.ci_95_upper < 0.0:
                if comp.p_value_holm is not None and comp.p_value_holm < alpha:
                    return True
        return False

    f1_pass, f1_details = _check_superiority(f1_comp)
    mcc_pass, mcc_details = _check_superiority(mcc_comp)

    # Check degradation on the other primary metric
    f1_degraded = _check_significant_degradation(f1_comp)
    mcc_degraded = _check_significant_degradation(mcc_comp)

    f1_superior = f1_pass and not mcc_degraded
    mcc_superior = mcc_pass and not f1_degraded

    if f1_pass and mcc_degraded:
        reasons.append("F1 passed criteria but MCC has statistically significant degradation")
    if mcc_pass and f1_degraded:
        reasons.append("MCC passed criteria but F1 has statistically significant degradation")

    is_superior = f1_superior or mcc_superior
    is_label_efficient = is_superior and fraction in {0.01, 0.05}

    if not is_superior:
        if not f1_pass:
            reasons.append(f"F1 illicit: {f1_details.get('fail_reason')}")
        if not mcc_pass:
            reasons.append(f"MCC: {mcc_details.get('fail_reason')}")

    return ClaimGateResult(
        comparison_id=comparison_id,
        fraction=fraction,
        is_f1_superior=f1_superior,
        is_mcc_superior=mcc_superior,
        is_superior=is_superior,
        is_label_efficient=is_label_efficient,
        f1_details=f1_details,
        mcc_details=mcc_details,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class AblationAttribution:
    """Quantitative attribution of representation and mechanism components at 1% label regime."""

    attribution_type: str
    target_variant: str
    reference_variant: str
    f1_differences: tuple[float, ...]
    mcc_differences: tuple[float, ...]
    f1_mean_difference: float
    f1_std_difference: float
    mcc_mean_difference: float
    mcc_std_difference: float
    f1_relative_gain_percent: float
    mcc_relative_gain_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_ablation_attribution(
    target_f1_by_seed: Mapping[int, float],
    target_mcc_by_seed: Mapping[int, float],
    reference_f1_by_seed: Mapping[int, float],
    reference_mcc_by_seed: Mapping[int, float],
    *,
    attribution_type: str,
    target_variant: str,
    reference_variant: str,
    required_seeds: Sequence[int] = CANONICAL_SEEDS,
) -> AblationAttribution:
    """Compute attribution difference and relative gain between target and reference."""
    f1_diffs = tuple(
        float(target_f1_by_seed[s] - reference_f1_by_seed[s])
        for s in required_seeds
        if s in target_f1_by_seed and s in reference_f1_by_seed
    )
    mcc_diffs = tuple(
        float(target_mcc_by_seed[s] - reference_mcc_by_seed[s])
        for s in required_seeds
        if s in target_mcc_by_seed and s in reference_mcc_by_seed
    )

    f1_mean = float(np.mean(f1_diffs)) if f1_diffs else 0.0
    f1_std = float(np.std(f1_diffs, ddof=1)) if len(f1_diffs) > 1 else 0.0
    mcc_mean = float(np.mean(mcc_diffs)) if mcc_diffs else 0.0
    mcc_std = float(np.std(mcc_diffs, ddof=1)) if len(mcc_diffs) > 1 else 0.0

    ref_f1_mean = float(np.mean([reference_f1_by_seed[s] for s in required_seeds if s in reference_f1_by_seed]))
    ref_mcc_mean = float(np.mean([reference_mcc_by_seed[s] for s in required_seeds if s in reference_mcc_by_seed]))

    f1_rel = (f1_mean / ref_f1_mean * 100.0) if ref_f1_mean > 1e-6 else 0.0
    mcc_rel = (mcc_mean / ref_mcc_mean * 100.0) if ref_mcc_mean > 1e-6 else 0.0

    return AblationAttribution(
        attribution_type=attribution_type,
        target_variant=target_variant,
        reference_variant=reference_variant,
        f1_differences=f1_diffs,
        mcc_differences=mcc_diffs,
        f1_mean_difference=f1_mean,
        f1_std_difference=f1_std,
        mcc_mean_difference=mcc_mean,
        mcc_std_difference=mcc_std,
        f1_relative_gain_percent=float(f1_rel),
        mcc_relative_gain_percent=float(mcc_rel),
    )


def compute_temporal_series(
    predictions_by_snapshot: Mapping[int, tuple[Sequence[int], Sequence[float]]],
    threshold: float,
) -> dict[int, dict[str, Any]]:
    """Compute independent per-snapshot performance series for time steps 35-49."""
    series: dict[int, dict[str, Any]] = {}
    for step in sorted(predictions_by_snapshot):
        labels, scores = predictions_by_snapshot[step]
        metrics = binary_metrics(labels, scores, threshold)
        series[step] = metrics
    return series
