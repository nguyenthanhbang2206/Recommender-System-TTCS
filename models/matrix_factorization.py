"""
Matrix Factorization (MF) sử dụng Stochastic Gradient Descent (SGD)
Đây là thuật toán lọc cộng tác cổ điển và hiệu quả.

Mô hình:
    R_hat[u, i] = mu + b_u + b_i + U[u] · V[i]^T

Trong đó:
    - mu   : rating trung bình toàn cục
    - b_u  : bias của user u
    - b_i  : bias của item i
    - U[u] : vector đặc trưng ẩn của user u (k chiều)
    - V[i] : vector đặc trưng ẩn của item i (k chiều)
"""
import numpy as np
import time


class MatrixFactorizationSGD:
    """
    Matrix Factorization với SGD, hỗ trợ bias và L2 regularization.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 50,
        lr: float = 0.005,
        reg: float = 0.02,
        n_epochs: int = 20,
        random_state: int = 42,
    ):
        """
        Args:
            n_users    : số lượng người dùng
            n_items    : số lượng sản phẩm/phim
            n_factors  : số chiều ẩn (latent factors)
            lr         : learning rate
            reg        : hệ số L2 regularization
            n_epochs   : số epoch huấn luyện
            random_state: seed ngẫu nhiên
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.train_losses = []
        self.val_losses = []

    def _init_params(self, global_mean: float):
        rng = np.random.RandomState(self.random_state)
        scale = 0.01
        self.global_mean = global_mean
        # Ma trận đặc trưng người dùng và sản phẩm
        self.U = rng.normal(0, scale, (self.n_users, self.n_factors))
        self.V = rng.normal(0, scale, (self.n_items, self.n_factors))
        # Bias
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

    def fit(self, train_data: np.ndarray, val_data: np.ndarray = None, verbose: bool = True):
        """
        Huấn luyện mô hình.

        Args:
            train_data: mảng numpy shape (N, 3) gồm [user_idx, item_idx, rating]
            val_data  : dữ liệu validation (tùy chọn)
            verbose   : hiển thị tiến độ
        """
        global_mean = train_data[:, 2].mean()
        self._init_params(global_mean)

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):
            # Xáo trộn dữ liệu mỗi epoch
            idx = np.random.permutation(len(train_data))
            samples = train_data[idx]

            epoch_loss = self._sgd_step(samples)
            self.train_losses.append(epoch_loss)

            val_rmse = ""
            if val_data is not None:
                vr = self._compute_rmse(val_data)
                self.val_losses.append(vr)
                val_rmse = f"| Val RMSE: {vr:.4f}"

            if verbose:
                elapsed = time.time() - start
                print(f"Epoch {epoch:>3}/{self.n_epochs} | Train Loss: {epoch_loss:.4f} {val_rmse} | Time: {elapsed:.1f}s")

        print(f"\n✓ Huấn luyện hoàn thành sau {time.time() - start:.1f}s")

    def _sgd_step(self, samples: np.ndarray) -> float:
        """Một bước SGD qua toàn bộ dữ liệu, trả về RMSE của epoch"""
        total_sq_err = 0.0
        for u, i, r in samples:
            u, i = int(u), int(i)
            # Dự đoán
            pred = self.global_mean + self.b_u[u] + self.b_i[i] + self.U[u] @ self.V[i]
            err = r - pred
            total_sq_err += err ** 2

            # Cập nhật bias
            self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
            self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

            # Cập nhật ma trận đặc trưng
            u_vec = self.U[u].copy()
            self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
            self.V[i] += self.lr * (err * u_vec   - self.reg * self.V[i])

        return np.sqrt(total_sq_err / len(samples))

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Dự đoán rating của user cho một item cụ thể"""
        pred = (
            self.global_mean
            + self.b_u[user_idx]
            + self.b_i[item_idx]
            + self.U[user_idx] @ self.V[item_idx]
        )
        # Clip về khoảng hợp lệ [1, 5]
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, pairs: np.ndarray) -> np.ndarray:
        """Dự đoán hàng loạt. pairs shape: (N, 2) gồm [user_idx, item_idx]"""
        users = pairs[:, 0].astype(int)
        items = pairs[:, 1].astype(int)
        preds = (
            self.global_mean
            + self.b_u[users]
            + self.b_i[items]
            + (self.U[users] * self.V[items]).sum(axis=1)
        )
        return np.clip(preds, 1.0, 5.0)

    def recommend(self, user_idx: int, n: int = 10, exclude_seen: set = None) -> list:
        """
        Gợi ý top-N sản phẩm cho một user.

        Args:
            user_idx    : chỉ số người dùng
            n           : số lượng gợi ý
            exclude_seen: tập item_idx đã tương tác (sẽ loại khỏi gợi ý)

        Returns:
            Danh sách (item_idx, predicted_rating) sắp xếp giảm dần theo rating dự đoán
        """
        scores = (
            self.global_mean
            + self.b_u[user_idx]
            + self.b_i
            + self.U[user_idx] @ self.V.T
        )
        scores = np.clip(scores, 1.0, 5.0)

        if exclude_seen:
            scores[list(exclude_seen)] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _compute_rmse(self, data: np.ndarray) -> float:
        """Tính RMSE trên một tập dữ liệu"""
        pairs = data[:, :2]
        y_true = data[:, 2]
        y_pred = self.predict_batch(pairs)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def save(self, path: str):
        """Lưu tham số mô hình"""
        np.savez(
            path,
            U=self.U, V=self.V,
            b_u=self.b_u, b_i=self.b_i,
            global_mean=np.array([self.global_mean]),
            config=np.array([self.n_users, self.n_items, self.n_factors])
        )
        print(f"✓ Đã lưu mô hình tại: {path}.npz")

    @classmethod
    def load(cls, path: str) -> "MatrixFactorizationSGD":
        """Tải mô hình đã lưu"""
        data = np.load(path + ".npz")
        config = data["config"].astype(int)
        model = cls(n_users=config[0], n_items=config[1], n_factors=config[2])
        model.U = data["U"]
        model.V = data["V"]
        model.b_u = data["b_u"]
        model.b_i = data["b_i"]
        model.global_mean = float(data["global_mean"][0])
        print(f"✓ Đã tải mô hình từ: {path}.npz")
        return model
