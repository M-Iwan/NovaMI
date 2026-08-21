import numpy as np
import numpy.typing as npt

from scipy.stats import mannwhitneyu
from novami.stats.utils import round_to_significant

def mwu_rank_biserial(u_stat: float, n1: int, n2: int):
    """
    Rank biserial effect size for Mann-Whitney U-test. Positive value correspond to values
    from population 1 being larger than from population 2.

    Parameters
    ----------
    u_stat : float
        Test statistic from scipy.stats.mannwhitneyu
    n1: int
        Size of population 1.
    n2: int
        Size of population 2.
    """

    return (2 * u_stat) / (n1 * n2) - 1


def mann_whitney_u_test(values_1: npt.NDArray, values_2: npt.NDArray, label_1: str, label_2: str,
                        min_n: int = 3, alpha: float = 0.05, alternative: str = "two-sided"):
    """
    Run a Mann-Whitney U test between two groups.

    Parameters
    ----------
    values_1 : npt.NDArray
        Samples from population 1.
    values_2 : npt.NDArray
        Samples from population 2.
    label_1 : str
        Name for population 1.
    label_2 : str
        Name for population 2.
    min_n : int, default=3
        Minimum sample size required per group.
    alpha : float, default=0.05
        Significance level.
    alternative : str, default="two-sided"
        Alternative hypothesis passed to scipy.stats.mannwhitneyu.
    """

    n1, n2 = len(values_1), len(values_2)
    null_hypothesis = f"{label_1} ~ {label_2}"
    low_n = (n1 < min_n) or (n2 < min_n)

    result = {
        "null_hypothesis": null_hypothesis,
        "label_1": label_1,
        "label_2": label_2,
        "n1": n1,
        "n2": n2,
        "mean_1": None,
        "mean_2": None,
        "median_1": None,
        "median_2": None,
        "u_stat": None,
        "p_value": None,
        "rbc": None,
        "effect_size": None,
        "low_n_flag": low_n,
        "outcome": None,
    }

    if low_n:
        print(f"Sample size too small for <{null_hypothesis}>. N1: {n1}, N2: {n2}")
        return result

    stat, p_value = mannwhitneyu(values_1, values_2, alternative=alternative)
    rbc = mwu_rank_biserial(stat, n1, n2)
    abs_rbc = abs(rbc)

    if abs_rbc <= 0.1:
        effect_size = "Negligible"
    elif abs_rbc <= 0.3:
        effect_size = "Small"
    elif abs_rbc <= 0.5:
        effect_size = "Medium"
    else:
        effect_size = "Large"

    if p_value < alpha:
        if rbc > 0:
            relation = ">"
        elif rbc < 0:
            relation = "<"
        else:
            relation = "~"
    else:
        relation = "~"

    result.update(
        {
            "mean_1": round_to_significant(values_1.mean(), 5),
            "mean_2": round_to_significant(values_2.mean(), 5),
            "median_1": round_to_significant(np.median(values_1), 5),
            "median_2": round_to_significant(np.median(values_2), 5),
            "u_stat": round_to_significant(stat, 5),
            "p_value": round_to_significant(p_value, 5),
            "rbc": round_to_significant(rbc, 5),
            "effect_size": effect_size,
            "outcome": f"{label_1} {relation} {label_2}",
        }
    )

    return result