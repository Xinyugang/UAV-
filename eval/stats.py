from __future__ import annotations

import itertools
from typing import Dict, List, Sequence
import numpy as np
from scipy.stats import rankdata, t as student_t


def exact_two_sided_sign_flip_p(diffs: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip p-value enumerating all 2^n sign vectors (Eq. S8).

    p = 2^{-n} sum_{z in {-1, +1}^n} 1{ |sum_g z_g d_g| >= |sum_g d_g| }.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n == 0:
        return 1.0
    obs = abs(float(np.sum(d)))
    if obs <= 1e-15:
        return 1.0

    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=np.float64)  # (2^n, n)
    flipped_sums = np.abs(signs @ d)
    # Small numerical tolerance to prevent floating-point loss on exact symmetry ties
    ge_count = int(np.sum(flipped_sums >= obs - 1e-12))
    return float(ge_count) / float(1 << n)


def holm_step_down_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down family-wise error adjustment (Supplementary Eq. below S8, S27-S28).

    p_adj_(i) = min(1, max_{j <= i} (m - j + 1) * p_(j)).
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = p_arr.size
    if m == 0:
        return []
    order = np.argsort(p_arr, kind="mergesort")
    sorted_p = p_arr[order]
    adj_sorted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for idx in range(m):
        scaled = (m - idx) * float(sorted_p[idx])
        running_max = max(running_max, scaled)
        adj_sorted[idx] = min(1.0, running_max)

    out = np.zeros(m, dtype=np.float64)
    out[order] = adj_sorted
    return out.tolist()


def matched_pairs_rank_biserial(diffs: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (W_+ - W_-) / (W_+ + W_-) with midranks (Table S12)."""
    d = np.asarray(diffs, dtype=np.float64)
    nz = d[np.abs(d) > 1e-15]
    if nz.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nz), method="average")
    w_pos = float(np.sum(ranks[nz > 0.0]))
    w_neg = float(np.sum(ranks[nz < 0.0]))
    denom = w_pos + w_neg
    return 0.0 if denom <= 0.0 else (w_pos - w_neg) / denom


def paired_t_summary(diffs: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Compute paired mean, sample SD, 95% t CI, Cohen's d_z, sign-flip p, and rank-biserial."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "sd": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "d_z": 0.0,
            "p_sign_flip": 1.0,
            "rank_biserial": 0.0,
        }
    mean = float(np.mean(d))
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "d_z": 0.0,
            "p_sign_flip": exact_two_sided_sign_flip_p(d),
            "rank_biserial": matched_pairs_rank_biserial(d),
        }

    sd = float(np.std(d, ddof=1))
    t_crit = float(student_t.ppf(1.0 - 0.5 * alpha, df=n - 1))
    half = t_crit * sd / float(np.sqrt(n))
    d_z = 0.0 if sd <= 1e-15 else mean / sd
    return {
        "n": float(n),
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "d_z": d_z,
        "p_sign_flip": exact_two_sided_sign_flip_p(d),
        "rank_biserial": matched_pairs_rank_biserial(d),
    }
