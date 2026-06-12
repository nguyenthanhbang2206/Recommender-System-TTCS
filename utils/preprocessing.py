"""
Hai chế độ:
  - Implicit feedback: convert rating → 0/1, leave-one-out split
  - Explicit feedback:   giữ rating 1-5, random/time split
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ═══════════════════════════════════════════════════════════
#  Encode user/item ID → 0-based index
# ═══════════════════════════════════════════════════════════

def encode_ids(ratings: pd.DataFrame):
    user2idx = {u: i for i, u in enumerate(sorted(ratings["userId"].unique()))}
    item2idx = {m: i for i, m in enumerate(sorted(ratings["movieId"].unique()))}
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    df = ratings.copy()
    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Encode xong: {n_users:,} users | {n_items:,} items")
    return df, user2idx, item2idx, idx2user, idx2item, n_users, n_items


# ═══════════════════════════════════════════════════════════
#  IMPLICIT — Leave-one-out split (đúng paper NCF)
# ═══════════════════════════════════════════════════════════

def leave_one_out_split(df: pd.DataFrame):
    df = df.sort_values(["user_idx", "timestamp"]).copy()

    test_rows  = df.groupby("user_idx").tail(1).index
    remain     = df.drop(index=test_rows)
    val_rows   = remain.groupby("user_idx").tail(1).index

    test_df  = df.loc[test_rows].reset_index(drop=True)
    val_df   = df.loc[val_rows].reset_index(drop=True)
    train_df = df.drop(index=test_rows.union(val_rows)).reset_index(drop=True)

    print(f"  Leave-one-out split:")
    print(f"    Train: {len(train_df):,} interactions")
    print(f"    Val  : {len(val_df):,} interactions (1/user)")
    print(f"    Test : {len(test_df):,} interactions (1/user)")
    return train_df, val_df, test_df


def convert_to_implicit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = 1
    return df


def build_negative_pool(train_df: pd.DataFrame, n_users: int, n_items: int,
                        extra_dfs: list = None):
    """
    Tạo tập negative candidates cho mỗi user.
    negative_pool[u] = items user u CHƯA tương tác trong TOÀN BỘ data
                       (train + val + test) để tránh sample nhầm test item.
    """
    all_dfs  = [train_df] + (extra_dfs or [])
    combined = pd.concat(all_dfs, ignore_index=True)

    # Tất cả items đã tương tác (kể cả val/test) → không được sample làm negative
    all_interacted = (
        combined.groupby("user_idx")["item_idx"]
        .apply(set).to_dict()
    )
    # Chỉ train interactions → dùng cho exclude_seen khi recommend
    interacted_train = (
        train_df.groupby("user_idx")["item_idx"]
        .apply(set).to_dict()
    )
    all_items = set(range(n_items))
    negative_pool = {
        u: list(all_items - all_interacted.get(u, set()))
        for u in range(n_users)
    }
    return negative_pool, interacted_train


# ═══════════════════════════════════════════════════════════
#  EXPLICIT — Random / Time-based split (dùng cho MF)
# ═══════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Random split cho explicit rating (MF baseline)."""
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size,
                                         random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_data_time(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    """Time-based split cho explicit rating (MF baseline)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end   = int(n * (1 - test_size))
    return (df.iloc[:train_end].reset_index(drop=True),
            df.iloc[train_end:val_end].reset_index(drop=True),
            df.iloc[val_end:].reset_index(drop=True))


# ═══════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════

def build_user_item_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """Tạo sparse user-item matrix từ interaction data."""
    return csr_matrix(
        (np.ones(len(df)), (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items)
    )


def get_statistics(ratings: pd.DataFrame, movies: pd.DataFrame):
    print(f"  Dataset: {len(ratings):,} ratings | "
          f"{ratings['userId'].nunique():,} users | "
          f"{ratings['movieId'].nunique():,} items")
    print(f"  Rating range: [{ratings['rating'].min()}, {ratings['rating'].max()}] "
          f"| Mean: {ratings['rating'].mean():.2f}")
    sparsity = 1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique())
    print(f"  Sparsity: {sparsity:.2%}")
