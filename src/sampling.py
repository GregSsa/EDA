"""Sampling utilities: weighted sampling and a light MCMC sampler.

Extracted from notebook code.
"""
from typing import Optional
import numpy as np
import pandas as pd


def weighted_sampling(df_pool: pd.DataFrame, k: int, with_replacement: bool = False, random_state: Optional[int] = None) -> pd.DataFrame:
    """Perform weighted sampling using column `final_sampling_weight`.
    Returns a DataFrame with the sampled rows (preserves index).
    """
    rng = np.random.default_rng(random_state)
    weights = df_pool['final_sampling_weight'].astype(float)
    total = weights.sum()
    if total <= 0 or np.isnan(total):
        # uniform sampling
        probs = None
    else:
        probs = (weights / total).to_numpy()

    indices = df_pool.index.to_numpy()
    if probs is None:
        chosen = rng.choice(indices, size=k, replace=with_replacement)
    else:
        chosen = rng.choice(indices, size=k, replace=with_replacement, p=probs)
    return df_pool.loc[chosen]


def light_mcmc(P_df: pd.DataFrame, k: int = 10, replace: bool = True, init_idx: Optional[int] = None,
               max_iters: int = 10000, random_state: Optional[int] = None) -> pd.DataFrame:
    """A simple MCMC-like sampler using weights in `final_sampling_weight`.

    Returns DataFrame of selected rows (reset index).
    """
    weights = P_df['final_sampling_weight'].to_numpy(dtype=float).copy()
    weights = np.maximum(weights, 0.0)
    if weights.sum() == 0:
        weights = np.ones_like(weights)

    rng = np.random.default_rng(random_state)
    n = len(weights)
    if n == 0:
        return P_df.copy()

    cur = int(rng.integers(n)) if init_idx is None else int(init_idx) % n
    samples = []
    visited = set()
    iters = 0

    while (len(samples) < k) and (iters < max_iters):
        prop = int(rng.integers(n))
        w_cur = weights[cur] if weights[cur] > 0 else 1e-12
        w_prop = weights[prop] if weights[prop] > 0 else 1e-12
        alpha = min(1.0, (w_prop / w_cur))
        if rng.random() < alpha:
            cur = prop

        if replace:
            samples.append(cur)
        else:
            if cur not in visited:
                samples.append(cur)
                visited.add(cur)
        iters += 1

    if (not replace) and (len(samples) < k):
        remaining = [i for i in range(n) if i not in set(samples)]
        if remaining:
            rem_weights = weights[remaining].astype(float)
            rem_weights = np.maximum(rem_weights, 0.0)
            if rem_weights.sum() == 0:
                rem_probs = np.ones(len(remaining)) / len(remaining)
            else:
                rem_probs = rem_weights / rem_weights.sum()
            need = k - len(samples)
            chosen = rng.choice(remaining, size=min(need, len(remaining)), replace=False, p=rem_probs)
            samples.extend(list(chosen))

    if len(samples) == 0:
        raise Exception("No samples could be collected.")

    return P_df.iloc[samples].reset_index(drop=True)
