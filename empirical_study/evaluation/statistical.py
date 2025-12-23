"""Statistical utilities for evaluation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats


def bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values to correct
        alpha: Significance level

    Returns:
        Tuple of (corrected alpha, list of significant indicators)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = [p < corrected_alpha for p in p_values]

    return corrected_alpha, significant


def holm_bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    Apply Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni while controlling FWER.

    Args:
        p_values: List of p-values to correct
        alpha: Significance level

    Returns:
        Tuple of (adjusted p-values, list of significant indicators)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Compute adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * (n - i)

    # Enforce monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Clip to 1
    adjusted = np.minimum(adjusted, 1.0)

    # Reorder to original
    result = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        result[idx] = adjusted[i]

    significant = [p < alpha for p in result]

    return result.tolist(), significant


def bootstrap_resample(
    data: np.ndarray,
    statistic_fn: callable,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Perform bootstrap resampling for any statistic.

    Args:
        data: Input data array
        statistic_fn: Function to compute statistic on data
        n_resamples: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with point estimate, CI bounds, and std error
    """
    np.random.seed(random_seed)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        try:
            stat = statistic_fn(sample)
            bootstrap_stats.append(stat)
        except Exception:
            continue

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence_level

    return {
        "point_estimate": statistic_fn(data),
        "mean": np.mean(bootstrap_stats),
        "std_error": np.std(bootstrap_stats),
        "ci_lower": np.percentile(bootstrap_stats, 100 * alpha / 2),
        "ci_upper": np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)),
    }


def paired_comparison(
    results_1: np.ndarray,
    results_2: np.ndarray,
    test: str = "mcnemar",
    correction: bool = True,
) -> Dict[str, float]:
    """
    Perform paired statistical comparison between two sets of results.

    Args:
        results_1: Results from method 1 (correct/incorrect or scores)
        results_2: Results from method 2
        test: Statistical test to use ('mcnemar', 'wilcoxon', 'ttest')
        correction: Apply correction where applicable

    Returns:
        Dictionary with test statistic and p-value
    """
    if test == "mcnemar":
        # For binary correct/incorrect data
        # Build 2x2 contingency table
        b = np.sum(results_1 & ~results_2)  # 1 correct, 2 wrong
        c = np.sum(~results_1 & results_2)  # 1 wrong, 2 correct

        if b + c == 0:
            return {"statistic": 0.0, "p_value": 1.0, "test": "mcnemar"}

        if correction:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
        else:
            statistic = (b - c) ** 2 / (b + c)

        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return {
            "statistic": statistic,
            "p_value": p_value,
            "test": "mcnemar",
            "b": int(b),
            "c": int(c),
        }

    elif test == "wilcoxon":
        # For paired continuous data
        try:
            statistic, p_value = stats.wilcoxon(results_1, results_2)
            return {
                "statistic": statistic,
                "p_value": p_value,
                "test": "wilcoxon",
            }
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {"statistic": 0.0, "p_value": 1.0, "test": "wilcoxon"}

    elif test == "ttest":
        # Paired t-test
        statistic, p_value = stats.ttest_rel(results_1, results_2)
        return {
            "statistic": statistic,
            "p_value": p_value,
            "test": "ttest_paired",
        }

    else:
        raise ValueError(f"Unknown test: {test}")


def compute_effect_size(
    group_1: np.ndarray,
    group_2: np.ndarray,
    method: str = "cohens_d",
) -> float:
    """
    Compute effect size between two groups.

    Args:
        group_1: First group data
        group_2: Second group data
        method: Effect size method ('cohens_d', 'hedges_g', 'glass_delta')

    Returns:
        Effect size value
    """
    mean_1 = np.mean(group_1)
    mean_2 = np.mean(group_2)

    if method == "cohens_d":
        # Pooled standard deviation
        n1, n2 = len(group_1), len(group_2)
        var1, var2 = np.var(group_1, ddof=1), np.var(group_2, ddof=1)
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        )
        if pooled_std == 0:
            return 0.0
        return (mean_1 - mean_2) / pooled_std

    elif method == "hedges_g":
        # Cohen's d with bias correction
        d = compute_effect_size(group_1, group_2, "cohens_d")
        n = len(group_1) + len(group_2)
        correction = 1 - (3 / (4 * n - 9))
        return d * correction

    elif method == "glass_delta":
        # Uses only control group std
        std_2 = np.std(group_2, ddof=1)
        if std_2 == 0:
            return 0.0
        return (mean_1 - mean_2) / std_2

    else:
        raise ValueError(f"Unknown method: {method}")


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Effect size value

    Returns:
        Interpretation string
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Perform chi-square test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies (uniform if None)

    Returns:
        Dictionary with statistic, p-value, and degrees of freedom
    """
    if expected is None:
        statistic, p_value = stats.chisquare(observed)
    else:
        statistic, p_value = stats.chisquare(observed, expected)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "df": len(observed) - 1,
    }


def compute_inter_annotator_agreement(
    annotations_1: np.ndarray,
    annotations_2: np.ndarray,
    method: str = "cohen_kappa",
) -> float:
    """
    Compute inter-annotator agreement.

    Args:
        annotations_1: First annotator's labels
        annotations_2: Second annotator's labels
        method: Agreement metric ('cohen_kappa', 'percent', 'fleiss_kappa')

    Returns:
        Agreement score
    """
    if method == "percent":
        return np.mean(annotations_1 == annotations_2)

    elif method == "cohen_kappa":
        from sklearn.metrics import cohen_kappa_score

        return cohen_kappa_score(annotations_1, annotations_2)

    else:
        raise ValueError(f"Unknown method: {method}")
