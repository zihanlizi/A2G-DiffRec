# partly adapted from the original metrics.py file in the FairDiffRec repository: https://github.com/danielemalitesta/FairDiffRec
from scipy import stats
import numba
import numpy as np
import math


@numba.jit(nopython=True)
def compute_DP(gr1_result, gr2_result):
    """Compute Demographic Parity (absolute difference between groups)"""
    return np.abs(gr1_result - gr2_result)


@numba.jit(nopython=True, parallel=True)
def compute_raw_exposure(topk_recs, mask, exposure_discount):
    """
    Compute exposure for items based on their positions in recommendation lists
    Using numba for performance optimization
    """
    exposure = np.zeros_like(mask, dtype=np.float32)
    items_ids = np.flatnonzero(mask)
    exp_disc_sum = (1 / exposure_discount).sum()

    for i in numba.prange(items_ids.shape[0]):
        item_id = items_ids[i]

        item_mask = np.zeros_like(mask, dtype=np.bool_)
        item_mask[item_id] = True

        item_presence = np.take(item_mask, topk_recs)
        item_exposure = item_presence / exposure_discount
        item_exposure = (item_exposure / exp_disc_sum).sum(axis=1).mean()
        exposure[item_id] = item_exposure

    return exposure

def compute_ndcg_per_user(predicted_indices, ground_truth, k=10):
    """
    Compute the Normalized Discounted Cumulative Gain (nDCG) for each user.
    """
    n_users = len(predicted_indices)
    ndcg_scores = np.zeros(n_users)

    for user_idx in range(n_users):
        if not ground_truth[user_idx]:
            continue

        # Compute DCG@k
        dcg = 0.0
        for rank, item_id in enumerate(predicted_indices[user_idx][:k], start=1):
            if item_id in ground_truth[user_idx]:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_rel_count = min(len(ground_truth[user_idx]), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_rel_count + 1))

        # Normalize
        ndcg_scores[user_idx] = dcg / idcg if idcg > 0 else 0.0

    return ndcg_scores



def compute_precision_per_user(predicted_indices, ground_truth, k):
    """Compute Precision for each user at cutoff k"""
    n_users = len(predicted_indices)
    precision_scores = np.zeros(n_users)

    for i in range(n_users):
        if len(ground_truth[i]) == 0:
            continue

        hits = 0
        for j in range(min(k, len(predicted_indices[i]))):
            if predicted_indices[i][j] in ground_truth[i]:
                hits += 1

        precision_scores[i] = hits / k

    return precision_scores


def compute_recall_per_user(predicted_indices, ground_truth, k):
    """Compute Recall for each user at cutoff k"""
    n_users = len(predicted_indices)
    recall_scores = np.zeros(n_users)

    for i in range(n_users):
        if len(ground_truth[i]) == 0:
            continue

        hits = 0
        for j in range(min(k, len(predicted_indices[i]))):
            if predicted_indices[i][j] in ground_truth[i]:
                hits += 1

        recall_scores[i] = hits / len(ground_truth[i])

    return recall_scores

# Coverage metrics
def compute_item_coverage(predicted_indices: np.ndarray, n_items: int, k: int) -> float:
    """
    Item Coverage@k: fraction of catalog that appears >=1 time in users' top-k.
    """
    # Get top-k items across all users
    topk = predicted_indices[:, :k].reshape(-1)
    unique_items = np.unique(topk)
    return float(unique_items.size) / float(n_items)

def compute_longtail_coverage(predicted_indices: np.ndarray, item_groups: np.ndarray, k: int) -> float:
    """
    Long-tail Coverage@k: fraction of LT items (item_groups==2) that appear >=1 time in top-k.
    """
    topk = predicted_indices[:, :k].reshape(-1)
    lt_items = np.flatnonzero(item_groups == 2)
    if lt_items.size == 0:
        return 0.0
    covered_lt = np.intersect1d(np.unique(topk), lt_items, assume_unique=False)
    return float(covered_lt.size) / float(lt_items.size)

def compute_item_popularity_distribution(predicted_indices: np.ndarray,
                                         n_items: int,
                                         k: int) -> np.ndarray:
    """
    Get the probability distribution of each item in the top-k recommendations.
    p(i) = number of times item i appears in the top-k recommendations / total number of pairs
    """
    topk = predicted_indices[:, :k].reshape(-1)

    counts = np.bincount(topk, minlength=n_items)
    total = counts.sum()
    if total == 0:
        return np.zeros(n_items, dtype=np.float64)
    return counts.astype(np.float64) / float(total)

def compute_gini_from_probs(p: np.ndarray) -> float:
    """
    Compute Gini index from the probability distribution p(i).
    Formula:
      Gini = 1/(N-1) * Σ_{k=1..N} (2k - N - 1) * p(i_k)
    where p(i_k) is the probability of item i_k sorted in ascending order.
    """
    N = p.size
    if N <= 1:
        return 0.0

    sorted_p = np.sort(p)
    k = np.arange(1, N + 1, dtype=np.float64)
    gini = np.sum((2 * k - N - 1) * sorted_p) / (N - 1)
    return float(gini)


def compute_gini_from_recs(predicted_indices: np.ndarray,
                           n_items: int,
                           k: int) -> float:
    """
    Compute Gini@K from the recommended items.
    """
    p = compute_item_popularity_distribution(predicted_indices, n_items, k)
    return compute_gini_from_probs(p)

def compute_shannon_entropy_from_probs(p: np.ndarray,
                                       base: float = 2.0) -> float:
    """
    Compute Shannon entropy from the probability distribution p(i).
      H = - Σ p(i) log_base(p(i))
    """
    mask = p > 0
    if not np.any(mask):
        return 0.0

    p_nonzero = p[mask]
    log_p = np.log(p_nonzero) / np.log(base)
    H = -np.sum(p_nonzero * log_p)
    return float(H)


def compute_shannon_entropy_from_recs(predicted_indices: np.ndarray,
                                      n_items: int,
                                      k: int,
                                      base: float = 2.0) -> float:
    """
    Compute Shannon entropy@K from the recommended items.
    """
    p = compute_item_popularity_distribution(predicted_indices, n_items, k)
    return compute_shannon_entropy_from_probs(p, base=base)


# Consumer-side fairness metrics
def compute_delta_ndcg(predicted_indices, ground_truth, user_groups, k):
    """
    Compute DeltaNDCG: absolute difference in NDCG between two user groups
    """
    ndcg_scores = compute_ndcg_per_user(predicted_indices, ground_truth, k)

    group1_mask = user_groups == 1
    group2_mask = user_groups == 2

    group1_ndcg = ndcg_scores[group1_mask].mean()
    group2_ndcg = ndcg_scores[group2_mask].mean()

    return compute_DP(group1_ndcg, group2_ndcg)


def compute_delta_precision(predicted_indices, ground_truth, user_groups, k):
    """Compute DeltaPrecision: absolute difference in Precision between two user groups"""
    precision_scores = compute_precision_per_user(predicted_indices, ground_truth, k)

    group1_mask = user_groups == 1
    group2_mask = user_groups == 2

    group1_precision = precision_scores[group1_mask].mean()
    group2_precision = precision_scores[group2_mask].mean()

    return compute_DP(group1_precision, group2_precision)


def compute_delta_recall(predicted_indices, ground_truth, user_groups, k):
    """Compute DeltaRecall: absolute difference in Recall between two user groups"""
    recall_scores = compute_recall_per_user(predicted_indices, ground_truth, k)

    group1_mask = user_groups == 1
    group2_mask = user_groups == 2

    group1_recall = recall_scores[group1_mask].mean()
    group2_recall = recall_scores[group2_mask].mean()

    return compute_DP(group1_recall, group2_recall)


def compute_delta_ndcg_pvalue(predicted_indices, ground_truth, user_groups, k, stat_test='mannwhitneyu'):
    """Compute statistical significance (p-value) of NDCG difference between groups"""
    ndcg_scores = compute_ndcg_per_user(predicted_indices, ground_truth, k)

    group1_mask = user_groups == 1
    group2_mask = user_groups == 2

    group1_scores = ndcg_scores[group1_mask]
    group2_scores = ndcg_scores[group2_mask]

    stat_func = getattr(stats, stat_test)
    result = stat_func(group1_scores, group2_scores)

    return result.pvalue


# Provider-side(Item-Side) fairness metrics

def compute_delta_exposure(predicted_indices, item_groups, short_head_ratio, k):
    """
    Compute DeltaExposure: absolute difference in exposure between short-head and long-tail items
    """
    topk_recs = predicted_indices[:, :k]
    exposure_discount = np.log2(np.arange(1, k + 1) + 1)

    sh_mask = item_groups == 1
    lt_mask = item_groups == 2

    sh_exposure = compute_raw_exposure(topk_recs, sh_mask, exposure_discount)
    lt_exposure = compute_raw_exposure(topk_recs, lt_mask, exposure_discount)

    # Normalize by group distribution
    group_distribution_sh = 1
    group_distribution_lt = 1 / short_head_ratio - 1

    sh_result = sh_exposure.sum() / group_distribution_sh
    lt_result = lt_exposure.sum() / group_distribution_lt

    return compute_DP(sh_result, lt_result)


def compute_delta_visibility(predicted_indices, item_groups, short_head_ratio, k):
    """
    Compute DeltaVisibility: absolute difference in visibility between short-head and long-tail items
    """
    topk_recs = predicted_indices[:, :k]
    n_items = len(item_groups)
    n_users = topk_recs.shape[0]

    # Count how many times each item appears
    flat_recs = topk_recs.flatten()
    raw_visibility = np.bincount(flat_recs, minlength=n_items)
    visibility_prob = raw_visibility / (n_users * k)

    sh_mask = item_groups == 1
    lt_mask = item_groups == 2

    # Normalize by group distribution
    group_distribution_sh = 1
    group_distribution_lt = 1 / short_head_ratio - 1

    sh_visibility = visibility_prob[sh_mask].sum() / group_distribution_sh
    lt_visibility = visibility_prob[lt_mask].sum() / group_distribution_lt

    return compute_DP(sh_visibility, lt_visibility)


def compute_aplt(predicted_indices, item_groups, k):
    """
    Compute APLT (Average Percentage of Long-Tail items):
    """
    topk_recs = predicted_indices[:, :k]
    n_items = len(item_groups)
    n_users = topk_recs.shape[0]

    # Count how many times each item appears
    flat_recs = topk_recs.flatten()
    raw_visibility = np.bincount(flat_recs, minlength=n_items)
    visibility_prob = raw_visibility / (n_users * k)

    lt_mask = item_groups == 2
    lt_visibility = visibility_prob[lt_mask].sum()

    return lt_visibility


def compute_all_consumer_metrics(predicted_indices, ground_truth, user_groups, topN, 
                                user_histories=None, popularity_bins=None):
    """
    Compute all consumer-side fairness metrics
    """
    results = {}

    for k in topN:
        results[f'DeltaNDCG@{k}'] = compute_delta_ndcg(predicted_indices, ground_truth, user_groups, k)
        results[f'DeltaPrecision@{k}'] = compute_delta_precision(predicted_indices, ground_truth, user_groups, k)
        results[f'DeltaRecall@{k}'] = compute_delta_recall(predicted_indices, ground_truth, user_groups, k)
        results[f'DeltaNDCG_pvalue@{k}'] = compute_delta_ndcg_pvalue(predicted_indices, ground_truth, user_groups, k)

    return results


def compute_all_provider_metrics(predicted_indices, item_groups, short_head_ratio, topN,
                                 n_items: int = None):
    """
    Compute all provider-side fairness metrics
    """
    results = {}

    for k in topN:
        results[f'DeltaExposure@{k}'] = compute_delta_exposure(predicted_indices, item_groups, short_head_ratio, k)
        results[f'DeltaVisibility@{k}'] = compute_delta_visibility(predicted_indices, item_groups, short_head_ratio, k)
        results[f'APLT@{k}'] = compute_aplt(predicted_indices, item_groups, k)
        n_items = len(item_groups)
        results[f'ItemCoverage@{k}'] = compute_item_coverage(predicted_indices, n_items, k)
        results[f'LongtailCoverage@{k}'] = compute_longtail_coverage(predicted_indices, item_groups, k)
        results[f'Gini@{k}'] = compute_gini_from_recs(
            predicted_indices, n_items, k
        )
        results[f'Entropy@{k}'] = compute_shannon_entropy_from_recs(
            predicted_indices, n_items, k
        )
    return results


def print_fairness_results(consumer_results=None, provider_results=None):
    """Print fairness evaluation results in a readable format"""
    out = ""

    if consumer_results is not None:
        out += "\n=== Consumer-side Fairness Metrics ===\n"
        for metric, value in consumer_results.items():
            out += f"{metric}: {value:.4f}\n"

    if provider_results is not None:
        out += "\n=== Provider-side Fairness Metrics ===\n"
        for metric, value in provider_results.items():
            out += f"{metric}: {value:.4f}\n"

    print(out)
    return out