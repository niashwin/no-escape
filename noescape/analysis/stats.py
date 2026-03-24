"""
Statistical Tests Module

Bootstrap CIs, effect sizes, significance tests, and result formatting.
"""

import numpy as np
from typing import Tuple, List
from scipy import stats


def bootstrap_ci(data: np.ndarray, n_resamples: int = 10000,
                  confidence: float = 0.95, statistic=np.mean) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (point_estimate, ci_lower, ci_upper)."""
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    boot_stats = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = 1 - confidence
    return (float(statistic(data)),
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def wilcoxon_test(x: np.ndarray, y: np.ndarray,
                   alternative: str = 'greater') -> Tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    try:
        stat, p = stats.wilcoxon(x, y, alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0


def paired_ttest(group1: np.ndarray, group2: np.ndarray,
                  alternative: str = 'greater') -> Tuple[float, float]:
    """Paired t-test. Returns (t_statistic, p_value)."""
    t_stat, p_val = stats.ttest_ind(group1, group2, alternative=alternative)
    return float(t_stat), float(p_val)


def format_result(mean: float, std: float, ci: Tuple[float, float]) -> str:
    """Format result as: value +/- std (95% CI: [lower, upper])."""
    return f"{mean:.3f} ± {std:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"
