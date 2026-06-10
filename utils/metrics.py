"""
Metrics đánh giá theo paper NCF (He et al. 2017):

  Implicit feedback evaluation (chuẩn paper):
    - leave-one-out: test item là interaction mới nhất của mỗi user
    - 100 negative samples + 1 test item → rank trong 101 items
    - HR@K  : Hit Ratio — test item có trong top-K không?
    - NDCG@K: Normalized DCG — vị trí của test item trong top-K

  Explicit feedback (MF baseline):
    - RMSE, MAE
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
#  Explicit feedback metrics (MF baseline)
# ═══════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ═══════════════════════════════════════════════════════════
#  Implicit feedback metrics (NCF — đúng paper)
# ═══════════════════════════════════════════════════════════

def hit_ratio_at_k(ranked_items: list, test_item: int, k: int) -> float:
    """
    HR@K: 1.0 nếu test_item xuất hiện trong top-K, ngược lại 0.0.
    (Đây là metric chính trong paper NCF)
    """
    return 1.0 if test_item in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items: list, test_item: int, k: int) -> float:
    """
    NDCG@K: log2(2) / log2(pos+2) nếu test_item trong top-K.
    Với leave-one-out chỉ có 1 relevant item nên IDCG = 1.
    (Metric chính thứ hai trong paper NCF)
    """
    if test_item in ranked_items[:k]:
        pos = ranked_items[:k].index(test_item)  # vị trí 0-based
        return 1.0 / np.log2(pos + 2)            # log2(1+1)=1 → NDCG=1 nếu rank 1
    return 0.0


def evaluate_implicit(
    model_score_fn,
    test_df: pd.DataFrame,
    negative_pool: dict,
    n_neg: int = 99,
    k: int = 10,
    seed: int = 42,
) -> dict:
    """
    Đánh giá implicit recommendation theo chuẩn paper NCF:
      - Với mỗi user: lấy 1 test item (positive) + n_neg negative items
      - Score tất cả 1 + n_neg items bằng model
      - Rank → tính HR@K và NDCG@K

    Args:
        model_score_fn : callable(users_arr, items_arr) → scores_arr
                         Nhận 2 numpy arrays shape (N,), trả về (N,) scores
        test_df        : DataFrame với cột [user_idx, item_idx]
                         Mỗi hàng là 1 positive test interaction (leave-one-out)
        negative_pool  : dict {user_idx: [item_idx, ...]} — items chưa tương tác
        n_neg          : số negative mẫu per user (paper dùng 99 → total 100)
        k              : cutoff (paper dùng 10)
        seed           : random seed cho reproducibility

    Returns:
        dict với HR@K, NDCG@K, và số user được đánh giá
    """
    rng = np.random.RandomState(seed)
    hr_list, ndcg_list = [], []

    for _, row in test_df.iterrows():
        user       = int(row["user_idx"])
        test_item  = int(row["item_idx"])

        # Sample n_neg negative items
        neg_pool = negative_pool.get(user, [])
        if len(neg_pool) == 0:
            continue
        n_sample   = min(n_neg, len(neg_pool))
        neg_items  = rng.choice(neg_pool, size=n_sample, replace=False).tolist()

        # Tạo candidate list: test_item + negatives
        candidates = [test_item] + neg_items                    # len = 1 + n_sample
        users_arr  = np.array([user] * len(candidates))
        items_arr  = np.array(candidates)

        # Score bằng model
        scores = model_score_fn(users_arr, items_arr)           # (len_candidates,)

        # Rank: argsort giảm dần, lấy item_idx tương ứng
        ranked_idx   = np.argsort(scores)[::-1]
        ranked_items = [candidates[i] for i in ranked_idx]

        hr_list.append(hit_ratio_at_k(ranked_items, test_item, k))
        ndcg_list.append(ndcg_at_k(ranked_items, test_item, k))

    return {
        f"HR@{k}":   float(np.mean(hr_list)),
        f"NDCG@{k}": float(np.mean(ndcg_list)),
        "n_users_evaluated": len(hr_list),
    }



def print_metrics(metrics: dict):
    print("\n  ─────────────────────────────────────")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"    {name:<25}: {value:.4f}")
        else:
            print(f"    {name:<25}: {value}")
    print("  ─────────────────────────────────────")
