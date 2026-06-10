"""
Unit tests — NCF (implicit) + MF (explicit) + metrics.
Chạy: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import (
    rmse, mae, hit_ratio_at_k, ndcg_at_k, evaluate_implicit,
)
from utils.preprocessing import (
    encode_ids, split_data, leave_one_out_split,
    convert_to_implicit, build_negative_pool,
)
from models.matrix_factorization import MatrixFactorizationSGD


# ── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def sample_ratings():
    np.random.seed(42)
    rows = []
    for user in range(1, 31):
        for movie in np.random.choice(range(1, 61), size=20, replace=False):
            rows.append({"userId": user, "movieId": int(movie),
                         "rating": float(np.random.choice([1,2,3,4,5])),
                         "timestamp": np.random.randint(900000000, 1000000000)})
    return pd.DataFrame(rows)

@pytest.fixture
def encoded(sample_ratings):
    df, u2i, i2i, i2u, i2item, n_u, n_i = encode_ids(sample_ratings)
    return df, n_u, n_i

@pytest.fixture
def implicit_splits(encoded):
    df, n_u, n_i = encoded
    df_impl = convert_to_implicit(df)
    train, val, test = leave_one_out_split(df_impl)
    neg_pool, interacted = build_negative_pool(train, n_u, n_i,
                                               extra_dfs=[val, test])
    return train, val, test, neg_pool, interacted, n_u, n_i


# ── Metrics (Implicit) ─────────────────────────────────────

class TestImplicitMetrics:
    def test_hr_hit(self):
        assert hit_ratio_at_k([3,1,2], 3, k=10) == 1.0

    def test_hr_hit_boundary(self):
        assert hit_ratio_at_k([1,2,3,4,5,6,7,8,9,10], 10, k=10) == 1.0

    def test_hr_miss(self):
        assert hit_ratio_at_k([1,2,3], 5, k=10) == 0.0

    def test_hr_outside_k(self):
        # item ở vị trí 5 (0-indexed) nhưng k=3 → miss
        assert hit_ratio_at_k([1,2,3,4,5,6], 5, k=3) == 0.0

    def test_ndcg_rank1(self):
        # rank 0 (vị trí đầu) → NDCG = 1/log2(2) = 1.0
        assert ndcg_at_k([3,1,2], 3, k=10) == pytest.approx(1.0)

    def test_ndcg_rank2(self):
        # rank 1 → NDCG = 1/log2(3)
        assert ndcg_at_k([1,3,2], 3, k=10) == pytest.approx(1/np.log2(3))

    def test_ndcg_miss(self):
        assert ndcg_at_k([1,2,4], 3, k=10) == 0.0

    def test_ndcg_outside_k(self):
        assert ndcg_at_k([1,2,3,4,5], 5, k=3) == 0.0


# ── Metrics (Explicit) ─────────────────────────────────────

class TestExplicitMetrics:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        assert rmse(np.array([3.0, 4.0]), np.array([4.0, 3.0])) == pytest.approx(1.0)

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        assert mae(np.array([3.0, 4.0]), np.array([5.0, 4.0])) == pytest.approx(1.0)


# ── Preprocessing ──────────────────────────────────────────

class TestPreprocessing:
    def test_encode_ids(self, sample_ratings):
        df, *_, n_u, n_i = encode_ids(sample_ratings)
        assert n_u == sample_ratings["userId"].nunique()
        assert n_i == sample_ratings["movieId"].nunique()
        assert df["user_idx"].min() == 0
        assert df["item_idx"].min() == 0

    def test_leave_one_out_sizes(self, encoded):
        df, n_u, n_i = encoded
        df_impl = convert_to_implicit(df)
        train, val, test = leave_one_out_split(df_impl)
        # Mỗi user có đúng 1 test và 1 val interaction
        assert test.groupby("user_idx").size().max() == 1
        assert val.groupby("user_idx").size().max() == 1

    def test_leave_one_out_no_overlap(self, encoded):
        df, n_u, n_i = encoded
        df_impl = convert_to_implicit(df)
        train, val, test = leave_one_out_split(df_impl)
        total = len(train) + len(val) + len(test)
        assert total == len(df_impl)

    def test_leave_one_out_latest(self, encoded):
        """Test item phải là interaction mới nhất của mỗi user."""
        df, n_u, n_i = encoded
        df_impl = convert_to_implicit(df)
        train, val, test = leave_one_out_split(df_impl)
        for _, row in test.iterrows():
            u = row["user_idx"]
            t_ts = row["timestamp"]
            # Tất cả train items của user này phải có timestamp nhỏ hơn
            train_ts = train[train["user_idx"]==u]["timestamp"]
            assert (train_ts <= t_ts).all()

    def test_negative_pool_no_overlap(self, implicit_splits):
        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits
        for u in range(n_u):
            if u in interacted and u in neg_pool:
                overlap = interacted[u] & set(neg_pool[u])
                assert len(overlap) == 0, f"User {u} có overlap giữa pos và neg pool"

    def test_convert_to_implicit(self, encoded):
        df, n_u, n_i = encoded
        impl = convert_to_implicit(df)
        assert "label" in impl.columns
        assert (impl["label"] == 1).all()


# ── evaluate_implicit ──────────────────────────────────────

class TestEvaluateImplicit:
    def test_perfect_model(self, implicit_splits):
        """Model luôn score test item cao nhất → HR@10 = 1.0."""
        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits

        def perfect_score(users, items):
            # Test item là phần tử đầu của candidate list
            test_items = test.set_index("user_idx")["item_idx"].to_dict()
            scores = np.zeros(len(users))
            for i, (u, item) in enumerate(zip(users, items)):
                if item == test_items.get(int(u), -1):
                    scores[i] = 1.0
            return scores

        res = evaluate_implicit(perfect_score, test.head(20), neg_pool,
                                n_neg=9, k=10, seed=42)
        assert res["HR@10"] == pytest.approx(1.0)

    def test_worst_model(self, implicit_splits):
        """
        Model luôn score test item thấp nhất → test item ở rank cuối.
        Dùng k=5 với n_neg=9 (10 candidates): test item ở rank 10 → miss k=5.
        """
        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits

        test_items_dict = test.set_index("user_idx")["item_idx"].to_dict()

        def worst_score(users, items):
            scores = np.ones(len(users))
            for i, (u, item) in enumerate(zip(users, items)):
                if item == test_items_dict.get(int(u), -1):
                    scores[i] = 0.0
            return scores

        # n_neg=9 → candidates=10, test item rank=10, k=5 → HR@5=0
        res = evaluate_implicit(worst_score, test.head(20), neg_pool,
                                n_neg=9, k=5, seed=42)
        assert res["HR@5"] == pytest.approx(0.0)

    def test_output_keys(self, implicit_splits):
        train, val, test, neg_pool, *_ = implicit_splits
        def dummy(u, i): return np.random.rand(len(u))
        res = evaluate_implicit(dummy, test.head(10), neg_pool, n_neg=9, k=10)
        assert "HR@10" in res
        assert "NDCG@10" in res
        assert 0.0 <= res["HR@10"] <= 1.0
        assert 0.0 <= res["NDCG@10"] <= 1.0


# ── MF (explicit baseline) ─────────────────────────────────

class TestMatrixFactorization:
    def test_predict_range(self, encoded):
        df, n_u, n_i = encoded
        train, test = split_data(df, test_size=0.2, random_state=42)
        arr = train[["user_idx","item_idx","rating"]].values
        mf = MatrixFactorizationSGD(n_u, n_i, n_factors=10, n_epochs=3)
        mf.fit(arr, verbose=False)
        pred = mf.predict(0, 0)
        assert 1.0 <= pred <= 5.0

    def test_loss_decreases(self, encoded):
        df, n_u, n_i = encoded
        arr = df[["user_idx","item_idx","rating"]].values
        mf = MatrixFactorizationSGD(n_u, n_i, n_factors=10, n_epochs=10)
        mf.fit(arr, verbose=False)
        assert mf.train_losses[-1] < mf.train_losses[0]

    def test_save_load(self, encoded, tmp_path):
        df, n_u, n_i = encoded
        arr = df[["user_idx","item_idx","rating"]].values
        mf = MatrixFactorizationSGD(n_u, n_i, n_factors=10, n_epochs=3)
        mf.fit(arr, verbose=False)
        path = str(tmp_path / "mf_test")
        mf.save(path)
        loaded = MatrixFactorizationSGD.load(path)
        assert mf.predict(0, 0) == pytest.approx(loaded.predict(0, 0), abs=1e-5)


# ── NCF (implicit) ─────────────────────────────────────────

class TestNeuralCF:
    def test_output_range(self, implicit_splits):
        try:
            import torch
            from models.neural_cf import NeuralCF
        except ImportError:
            pytest.skip("PyTorch không có")

        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits
        ncf = NeuralCF(n_u, n_i, mf_dim=8, mlp_layers=[16,8], n_epochs=1,
                       batch_size=64, seed=42)
        ncf.fit(train, interacted, verbose=False)

        users = np.array([0, 1, 2])
        items = np.array([0, 1, 2])
        scores = ncf.score(users, items)
        assert all(0.0 <= s <= 1.0 for s in scores), "Scores phải trong [0,1]"

    def test_recommend_exclude_seen(self, implicit_splits):
        try:
            import torch
            from models.neural_cf import NeuralCF
        except ImportError:
            pytest.skip("PyTorch không có")

        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits
        ncf = NeuralCF(n_u, n_i, mf_dim=8, mlp_layers=[16,8], n_epochs=1,
                       batch_size=64, seed=42)
        ncf.fit(train, interacted, verbose=False)

        seen = interacted.get(0, set())
        recs = ncf.recommend(0, n=5, exclude_seen=seen)
        rec_items = {item for item, _ in recs}
        assert len(rec_items & seen) == 0, "Không được gợi ý item đã tương tác"

    def test_save_load(self, implicit_splits, tmp_path):
        try:
            import torch
            from models.neural_cf import NeuralCF
        except ImportError:
            pytest.skip("PyTorch không có")

        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits
        ncf = NeuralCF(n_u, n_i, mf_dim=8, mlp_layers=[16,8], n_epochs=1,
                       batch_size=64, seed=42)
        ncf.fit(train, interacted, verbose=False)
        path = str(tmp_path / "ncf_test.pt")
        ncf.save(path)
        loaded = NeuralCF.load(path)

        users = np.array([0, 1])
        items = np.array([0, 1])
        orig   = ncf.score(users, items)
        loaded_ = loaded.score(users, items)
        np.testing.assert_allclose(orig, loaded_, atol=1e-5)



class TestPretraining:
    def test_gmf_standalone(self, implicit_splits):
        try:
            import torch
            from models.neural_cf import GMF, ImplicitDataset, _train_standalone
        except ImportError:
            pytest.skip("PyTorch khong co")
        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits
        gmf = GMF(n_u, n_i, mf_dim=8)
        pos_u = train["user_idx"].values.astype(int)
        pos_i = train["item_idx"].values.astype(int)
        ds = ImplicitDataset(pos_u, pos_i, n_i, interacted, n_neg=2)
        gmf = _train_standalone(gmf, ds, 2, 64, 1e-3, "cpu", False, "GMF")
        u = torch.LongTensor([0, 1]); i = torch.LongTensor([0, 1])
        scores = gmf(u, i).detach().numpy()
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_mlp_standalone(self, implicit_splits):
        try:
            import torch
            from models.neural_cf import MLP, ImplicitDataset, _train_standalone
        except ImportError:
            pytest.skip("PyTorch khong co")
        train, val, test, neg_pool, interacted, n_u, n_i = implicit_splits
        mlp = MLP(n_u, n_i, mlp_layers=[16, 8], dropout=0.1)
        pos_u = train["user_idx"].values.astype(int)
        pos_i = train["item_idx"].values.astype(int)
        ds = ImplicitDataset(pos_u, pos_i, n_i, interacted, n_neg=2)
        mlp = _train_standalone(mlp, ds, 2, 64, 1e-3, "cpu", False, "MLP")
        u = torch.LongTensor([0, 1]); i = torch.LongTensor([0, 1])
        scores = mlp(u, i).detach().numpy()
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_init_from_pretrained(self, implicit_splits):
        try:
            import torch
            from models.neural_cf import GMF, MLP, NeuMF, ImplicitDataset, _train_standalone
        except ImportError:
            pytest.skip("PyTorch khong co")
        train, _, _, _, interacted, n_u, n_i = implicit_splits
        gmf = GMF(n_u, n_i, mf_dim=8)
        mlp = MLP(n_u, n_i, mlp_layers=[16, 8], dropout=0.1)
        pos_u = train["user_idx"].values.astype(int)
        pos_i = train["item_idx"].values.astype(int)
        ds = ImplicitDataset(pos_u, pos_i, n_i, interacted, n_neg=2)
        gmf = _train_standalone(gmf, ds, 2, 64, 1e-3, "cpu", False, "GMF")
        mlp = _train_standalone(mlp, ds, 2, 64, 1e-3, "cpu", False, "MLP")
        neumf = NeuMF(n_u, n_i, mf_dim=8, mlp_layers=[16, 8], dropout=0.1)
        neumf.init_from_pretrained(gmf, mlp, alpha=0.5)
        u = torch.LongTensor([0]); i = torch.LongTensor([0])
        score = neumf(u, i).item()
        assert 0.0 <= score <= 1.0

    def test_neuralcf_with_pretrain(self, implicit_splits):
        try:
            from models.neural_cf import NeuralCF
        except ImportError:
            pytest.skip("PyTorch khong co")
        train, _, _, _, interacted, n_u, n_i = implicit_splits
        ncf = NeuralCF(n_u, n_i, mf_dim=8, mlp_layers=[16, 8],
                       dropout=0.1, n_epochs=2, batch_size=64, seed=42,
                       use_pretrain=True, pretrain_epochs=2)
        ncf.fit(train, interacted, verbose=False)
        scores = ncf.score(np.array([0, 1]), np.array([0, 1]))
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_neuralcf_without_pretrain(self, implicit_splits):
        try:
            from models.neural_cf import NeuralCF
        except ImportError:
            pytest.skip("PyTorch khong co")
        train, _, _, _, interacted, n_u, n_i = implicit_splits
        ncf = NeuralCF(n_u, n_i, mf_dim=8, mlp_layers=[16, 8],
                       dropout=0.1, n_epochs=2, batch_size=64, seed=42,
                       use_pretrain=False)
        ncf.fit(train, interacted, verbose=False)
        scores = ncf.score(np.array([0, 1]), np.array([0, 1]))
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_eval_mode_deterministic(self, implicit_splits):
        """Eval mode: dropout inactive, output phai deterministic."""
        try:
            import torch
            from models.neural_cf import NeuMF
        except ImportError:
            pytest.skip("PyTorch khong co")
        _, _, _, _, _, n_u, n_i = implicit_splits
        model = NeuMF(n_u, n_i, mf_dim=8, mlp_layers=[16, 8], dropout=0.5)
        u = torch.LongTensor([0]); i = torch.LongTensor([0])
        model.eval()
        with torch.no_grad():
            s1 = model(u, i).item()
            s2 = model(u, i).item()
        assert s1 == pytest.approx(s2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

