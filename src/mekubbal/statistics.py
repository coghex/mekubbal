from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


def bootstrap_mean_confidence(
    values: Any,
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
    empty_message: str = "Cannot compute confidence metrics from empty values.",
    include_sign_probabilities: bool = False,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(empty_message)

    mean_value = float(array.mean())
    if array.size == 1:
        result = {
            "mean": mean_value,
            "ci_low": mean_value,
            "ci_high": mean_value,
            "ci_width": 0.0,
        }
        if include_sign_probabilities:
            result["p_gt_zero"] = 1.0 if mean_value > 0 else 0.0 if mean_value < 0 else 0.5
            result["p_lt_zero"] = 1.0 if mean_value < 0 else 0.0 if mean_value > 0 else 0.5
        return result

    rng = np.random.default_rng(seed)
    sample_size = int(array.size)
    indices = rng.integers(0, sample_size, size=(int(n_bootstrap), sample_size))
    boot_means = array[indices].mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    ci_low = float(np.quantile(boot_means, alpha))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha))
    result = {
        "mean": mean_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
    }
    if include_sign_probabilities:
        result["p_gt_zero"] = float((boot_means > 0).mean())
        result["p_lt_zero"] = float((boot_means < 0).mean())
    return result


def paired_permutation_stats(
    differences: Any,
    *,
    n_permutation: int,
    seed: int,
    empty_message: str = "Cannot compute paired significance on empty differences.",
) -> dict[str, float]:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError(empty_message)

    observed = float(diffs.mean())
    pair_count = int(diffs.size)

    if pair_count <= 16:
        signs = np.asarray(list(product([-1.0, 1.0], repeat=pair_count)), dtype=float)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(int(n_permutation), pair_count))
    permutation_means = (signs * diffs).mean(axis=1)

    return {
        "mean_diff": observed,
        "pair_count": float(pair_count),
        "p_two_sided": float((np.abs(permutation_means) >= abs(observed)).mean()),
        "p_observed_or_higher": float((permutation_means >= observed).mean()),
        "p_observed_or_lower": float((permutation_means <= observed).mean()),
    }
