"""HASA: train-only heterogeneity score (paper Sec. IV-A) and budgeted width allocation
(Sec. IV-B, Algorithm 1), plus the alternative allocation policies used as baselines."""
from collections import Counter

import numpy as np


# ---------------------------------------------------------------- heterogeneity score
def js_divergence(p, q):
    """Jensen-Shannon divergence in nats (0 .. ln 2)."""
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def heterogeneity_scores(bench, alpha=0.1):
    """h_i = JSD(p_i || p_g): each client's smoothed token distribution vs the pooled
    distribution of all clients' TRAINING titles. Uses training data only."""
    df, tok = bench.df, bench.tokenizer
    all_train_ids = set().union(*(c.train_article_ids for c in bench.clients))
    global_counts = Counter(t for title in df[df.index.isin(all_train_ids)]["cleaned_title"] for t in tok(title))
    vocab = sorted(global_counts)

    def probs(counts):
        denom = sum(counts.values()) + alpha * len(vocab)
        return np.array([(counts.get(t, 0) + alpha) / denom for t in vocab], dtype=np.float64)

    p_global = probs(global_counts)
    scores = []
    for c in bench.clients:
        titles = df[df.index.isin(c.train_article_ids)]["cleaned_title"].tolist()
        local = Counter(t for title in titles for t in tok(title) if title)
        scores.append(float(js_divergence(probs(local), p_global)) if local else 0.0)
    return scores


# ---------------------------------------------------------------- allocation
def normalize_01(x):
    x = np.array(x, dtype=float)
    if x.max() > x.min():
        return (x - x.min()) / (x.max() - x.min())
    return np.ones_like(x) * 0.5


def normalize_rank_01(x):
    """(rank - 1) / (K - 1): robust to outliers."""
    x = np.array(x, dtype=float)
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, num=len(x))
    return ranks


def enforce_weighted_budget(ratios, weights, target_weighted_mean):
    """Scale all widths so that sum_i (n_i / N) r_i = B."""
    ratios = np.array(ratios, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    current = np.sum(ratios * weights)
    if current == 0:
        return ratios.tolist()
    return (ratios * (target_weighted_mean / current)).tolist()


def allocate_subnet_ratios(client_sizes, scores, policy, min_ratio=0.2, max_ratio=0.8, alpha=0.5,
                           invert_difficulty=False, enforce_budget=True, target_weighted_mean=None):
    """Width ratio per client.

    policy: "difficulty_only" = HASA, "uniform_50", "size_only", "mixed" (alpha*size + (1-alpha)*het),
    "fedavg_full" (every client trains the full model). invert_difficulty=True gives the
    inverse ablation (larger widths to LESS heterogeneous clients).
    """
    sizes_n = normalize_01(client_sizes)
    het_n = normalize_rank_01(scores)

    if policy == "uniform_50":
        return [0.5] * len(client_sizes)
    if policy == "fedavg_full":
        return [1.0] * len(client_sizes)
    if policy == "size_only":
        score = sizes_n
    elif policy == "difficulty_only":
        score = (1.0 - het_n) if invert_difficulty else het_n
    elif policy == "mixed":
        d = (1.0 - het_n) if invert_difficulty else het_n
        score = alpha * sizes_n + (1.0 - alpha) * d
    else:
        raise ValueError(f"Unknown policy: {policy}")

    ratios = min_ratio + np.clip(score, 0.0, 1.0) * (max_ratio - min_ratio)
    if enforce_budget:
        if target_weighted_mean is None:
            target_weighted_mean = (min_ratio + max_ratio) / 2
        for _ in range(2):  # Algorithm 1 with T = 2: scale to the budget, then project to [r_min, r_max]
            ratios = enforce_weighted_budget(ratios, client_sizes, target_weighted_mean)
            ratios = np.clip(ratios, min_ratio, max_ratio)
    return ratios.tolist()


def hasa_allocation(client_sizes, scores, budget=0.5, min_ratio=0.2, max_ratio=0.8):
    """HASA (Algorithm 1): rank-normalise the heterogeneity scores, map them linearly to
    [r_min, r_max], then rescale and clip until the size-weighted mean width equals the budget."""
    return allocate_subnet_ratios(client_sizes, scores, "difficulty_only", min_ratio, max_ratio,
                                  target_weighted_mean=budget)
