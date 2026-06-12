"""
  - Implicit feedback: label = 1 (observed) / 0 (sampled negative)
  - Loss: Binary Cross-Entropy (log loss)
  - Negative sampling: 4 negatives per positive (paper setting)
  - Output: Sigmoid → [0, 1] (xác suất tương tác)
  - Evaluation: HR@10 + NDCG@10 trên 1 test item + 99 sampled negatives
  - Pre-training: GMF → MLP → khởi tạo NeuMF
"""
import copy
import random
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
#  Dataset với Negative Sampling động
# ═══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class ImplicitDataset(Dataset):
        """
        Dataset cho implicit feedback với dynamic negative sampling.
        Mỗi epoch _resample() được gọi để tạo negatives mới.
        """
        def __init__(self, pos_users, pos_items, n_items, interacted, n_neg=4):
            self.pos_users  = pos_users
            self.pos_items  = pos_items
            self.n_items    = n_items
            self.interacted = interacted
            self.n_neg      = n_neg
            self._resample()

        def _resample(self):
            users, items, labels = [], [], []
            for u, i in zip(self.pos_users, self.pos_items):
                users.append(u); items.append(i); labels.append(1.0)
                seen  = self.interacted.get(int(u), set())
                count = 0
                while count < self.n_neg:
                    j = random.randint(0, self.n_items - 1)
                    if j not in seen:
                        users.append(u); items.append(j); labels.append(0.0)
                        count += 1
            self.users  = torch.LongTensor(users)
            self.items  = torch.LongTensor(items)
            self.labels = torch.FloatTensor(labels)

        def __len__(self):  return len(self.labels)
        def __getitem__(self, idx): return self.users[idx], self.items[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════
#  GMF — Generalized Matrix Factorization
# ═══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class GMF(nn.Module):
        """
        Generalized Matrix Factorization.
        Standalone model để pre-train trước khi ghép vào NeuMF.
        """
        def __init__(self, n_users: int, n_items: int, mf_dim: int = 32):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, mf_dim)
            self.item_emb = nn.Embedding(n_items, mf_dim)
            self.output   = nn.Linear(mf_dim, 1)
            nn.init.normal_(self.user_emb.weight, 0, 0.01)
            nn.init.normal_(self.item_emb.weight, 0, 0.01)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, user_ids, item_ids):
            gmf_out = self.user_emb(user_ids) * self.item_emb(item_ids)
            return torch.sigmoid(self.output(gmf_out).squeeze(-1))


# ═══════════════════════════════════════════════════════════
#  MLP — Multi-Layer Perceptron standalone
# ═══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """
        MLP model standalone.
        Pre-train trước khi ghép vào NeuMF.
        """
        def __init__(self, n_users, n_items, mlp_layers=None, dropout=0.1):
            super().__init__()
            mlp_layers   = mlp_layers or [64, 32, 16, 8]
            mlp_emb_dim  = mlp_layers[0] // 2
            self.user_emb = nn.Embedding(n_users, mlp_emb_dim)
            self.item_emb = nn.Embedding(n_items, mlp_emb_dim)

            layers = []
            in_sz  = mlp_layers[0]
            for out_sz in mlp_layers[1:]:
                layers += [nn.Linear(in_sz, out_sz), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                in_sz = out_sz
            self.mlp    = nn.Sequential(*layers)
            self.output = nn.Linear(mlp_layers[-1], 1)

            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, 0, 0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, user_ids, item_ids):
            mlp_in  = torch.cat([self.user_emb(user_ids),
                                  self.item_emb(item_ids)], dim=-1)
            mlp_out = self.mlp(mlp_in)
            return torch.sigmoid(self.output(mlp_out).squeeze(-1))


# ═══════════════════════════════════════════════════════════
#  NeuMF — Neural Matrix Factorization (GMF + MLP fused)
# ═══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class NeuMF(nn.Module):
        """
        NeuMF = GMF branch + MLP branch, concat → output.
        Có thể khởi tạo từ pre-trained GMF và MLP weights.
        """
        def __init__(self, n_users, n_items, mf_dim=32,
                     mlp_layers=None, dropout=0.1):
            super().__init__()
            mlp_layers  = mlp_layers or [64, 32, 16, 8]
            self.mf_dim     = mf_dim
            self.mlp_layers = mlp_layers

            # GMF branch — embedding riêng
            self.gmf_user_emb = nn.Embedding(n_users, mf_dim)
            self.gmf_item_emb = nn.Embedding(n_items, mf_dim)

            # MLP branch — embedding riêng
            mlp_emb_dim = mlp_layers[0] // 2
            self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
            self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)

            # MLP tower: halving pattern, ReLU, Dropout
            mlp_seq = []
            in_size = mlp_layers[0]
            for out_size in mlp_layers[1:]:
                mlp_seq.append(nn.Linear(in_size, out_size))
                mlp_seq.append(nn.ReLU())
                if dropout > 0:
                    mlp_seq.append(nn.Dropout(p=dropout))
                in_size = out_size
            self.mlp = nn.Sequential(*mlp_seq)

            # Output: concat(GMF, MLP) → Linear(→1) → Sigmoid
            self.output_layer = nn.Linear(mf_dim + mlp_layers[-1], 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def init_from_pretrained(self, gmf_model: "GMF", mlp_model: "MLP",
                                  alpha: float = 0.5):
            """
            Khởi tạo NeuMF từ pre-trained GMF và MLP.
            h ← [α * h_GMF, (1-α) * h_MLP] — paper dùng α=0.5.
            """
            # Copy GMF embeddings
            self.gmf_user_emb.weight.data.copy_(gmf_model.user_emb.weight.data)
            self.gmf_item_emb.weight.data.copy_(gmf_model.item_emb.weight.data)
            # Copy MLP embeddings
            self.mlp_user_emb.weight.data.copy_(mlp_model.user_emb.weight.data)
            self.mlp_item_emb.weight.data.copy_(mlp_model.item_emb.weight.data)
            # Copy MLP tower weights
            mlp_linears_src  = [m for m in mlp_model.mlp.modules() if isinstance(m, nn.Linear)]
            mlp_linears_dst  = [m for m in self.mlp.modules()       if isinstance(m, nn.Linear)]
            for src, dst in zip(mlp_linears_src, mlp_linears_dst):
                dst.weight.data.copy_(src.weight.data)
                dst.bias.data.copy_(src.bias.data)
            # Output layer: concat h_GMF * α và h_MLP * (1-α)
            gmf_h = gmf_model.output.weight.data          # (1, mf_dim)
            mlp_h = mlp_model.output.weight.data          # (1, mlp_layers[-1])
            combined_h = torch.cat([alpha * gmf_h, (1 - alpha) * mlp_h], dim=1)
            self.output_layer.weight.data.copy_(combined_h)
            bias = alpha * gmf_model.output.bias.data + (1 - alpha) * mlp_model.output.bias.data
            self.output_layer.bias.data.copy_(bias)

        def forward(self, user_ids, item_ids):
            gmf_out  = self.gmf_user_emb(user_ids) * self.gmf_item_emb(item_ids)
            mlp_in   = torch.cat([self.mlp_user_emb(user_ids),
                                   self.mlp_item_emb(item_ids)], dim=-1)
            mlp_out  = self.mlp(mlp_in)
            combined = torch.cat([gmf_out, mlp_out], dim=-1)
            return torch.sigmoid(self.output_layer(combined).squeeze(-1))


# ═══════════════════════════════════════════════════════════
#  Helper: train một model standalone (GMF hoặc MLP)
# ═══════════════════════════════════════════════════════════

def _train_standalone(model, dataset, n_epochs, batch_size, lr,
                      device, verbose, label):
    """Train GMF hoặc MLP standalone cho pre-training."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.to(device)
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        dataset._resample()
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
        model.train()
        total = 0.0
        for users, items, labels in loader:
            users  = users.to(device)
            items  = items.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        avg = total / len(dataset)
        if verbose:
            print(f"  [{label}] Epoch {epoch:>2}/{n_epochs} | BCE: {avg:.4f} | {time.time()-start:.1f}s")
    return model


# ═══════════════════════════════════════════════════════════
#  NeuralCF Wrapper
# ═══════════════════════════════════════════════════════════

class NeuralCF:
    """
    Tính năng:
      - BCE loss (log loss), implicit feedback 0/1
      - Dynamic negative sampling (4 neg/pos, resample mỗi epoch)
      - Pre-training: GMF + MLP standalone → khởi tạo NeuMF
      - Dropout 0.1 trong MLP (giảm overfitting trên sparse data)
      - Early stopping theo Val HR@10 ↑, restore best weights
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 32,
        mlp_layers: list = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_epochs: int = 20,
        batch_size: int = 256,
        n_neg: int = 4,
        device: str = None,
        seed: int = 42,
        early_stopping_patience: int = 5,
        min_delta: float = 1e-4,
        pretrain_epochs: int = 10,
        use_pretrain: bool = True,
        alpha: float = 0.5,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch chưa được cài. Chạy: pip install torch")

        self.n_users    = n_users
        self.n_items    = n_items
        self.mf_dim     = mf_dim
        self.mlp_layers = mlp_layers or [64, 32, 16, 8]
        self.dropout    = dropout
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.n_neg      = n_neg
        self.seed       = seed
        self.early_stopping_patience = early_stopping_patience
        self.min_delta  = min_delta
        self.pretrain_epochs = pretrain_epochs
        self.use_pretrain    = use_pretrain
        self.alpha           = alpha
        self.lr              = lr
        self.weight_decay    = weight_decay

        self.train_losses = []
        self.val_hr       = []

        set_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NeuMF(
            n_users=n_users, n_items=n_items,
            mf_dim=mf_dim, mlp_layers=self.mlp_layers, dropout=dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.criterion = nn.BCELoss()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ NeuMF | device={self.device} | mf_dim={mf_dim} | "
              f"mlp={self.mlp_layers} | dropout={dropout} | "
              f"neg={n_neg} | pretrain={use_pretrain} | params={n_params:,}")

    # ─────────────────────────────────────────
    #  Pre-training
    # ─────────────────────────────────────────

    def _pretrain(self, dataset: "ImplicitDataset", verbose: bool):
        """
        Pre-train GMF và MLP standalone, sau đó
        dùng weights của chúng để khởi tạo NeuMF.
        """
        print(f"\n  [Pre-training] Train GMF ({self.pretrain_epochs} epochs)...")
        gmf = GMF(self.n_users, self.n_items, self.mf_dim)
        _train_standalone(gmf, dataset, self.pretrain_epochs,
                          self.batch_size, self.lr, self.device, verbose, "GMF")

        print(f"\n  [Pre-training] Train MLP ({self.pretrain_epochs} epochs)...")
        mlp = MLP(self.n_users, self.n_items, self.mlp_layers, self.dropout)
        _train_standalone(mlp, dataset, self.pretrain_epochs,
                          self.batch_size, self.lr, self.device, verbose, "MLP")

        print(f"\n  [Pre-training] Khởi tạo NeuMF từ GMF + MLP weights (α={self.alpha})...")
        self.model.init_from_pretrained(gmf, mlp, alpha=self.alpha)

        # Reset optimizer (Adam momentum không hợp lệ với pre-trained weights)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr * 0.1,   # lr nhỏ hơn cho fine-tuning
            momentum=0.9,
        )
        print("  [Pre-training] Xong. Fine-tune NeuMF với SGD...")

    # ─────────────────────────────────────────
    #  Training epoch
    # ─────────────────────────────────────────

    def _train_epoch(self, dataset: "ImplicitDataset") -> float:
        dataset._resample()
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=0)
        self.model.train()
        total_loss = 0.0
        for users, items, labels in loader:
            users  = users.to(self.device)
            items  = items.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(users, items)
            loss  = self.criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(dataset)

    @torch.no_grad()
    def _score_batch(self, users_arr, items_arr):
        self.model.eval()
        u = torch.LongTensor(users_arr).to(self.device)
        i = torch.LongTensor(items_arr).to(self.device)
        return self.model(u, i).cpu().numpy()

    # ─────────────────────────────────────────
    #  fit
    # ─────────────────────────────────────────

    def fit(self, train_df, interacted: dict,
            val_eval_fn=None, verbose: bool = True):
        """
        Huấn luyện NeuMF với implicit feedback + negative sampling.
        Nếu use_pretrain=True: pre-train GMF và MLP trước, rồi fine-tune NeuMF.

        Args:
            train_df    : DataFrame [user_idx, item_idx, label=1]
            interacted  : {user_idx: set(item_idx)} từ train
            val_eval_fn : callable() → {"HR@10": float, "NDCG@10": float}
        """
        pos_users = train_df["user_idx"].values.astype(int)
        pos_items = train_df["item_idx"].values.astype(int)
        dataset   = ImplicitDataset(pos_users, pos_items,
                                    self.n_items, interacted, self.n_neg)

        if self.use_pretrain:
            self._pretrain(dataset, verbose=False)

        best_hr      = -1.0
        best_weights = None
        patience_ctr = 0
        start        = time.time()

        print(f"\n  [NeuMF fine-tune] {self.n_epochs} epochs...")
        for epoch in range(1, self.n_epochs + 1):
            bce = self._train_epoch(dataset)
            self.train_losses.append(bce)

            val_str    = ""
            val_hr_now = None
            if val_eval_fn is not None:
                val_metrics = val_eval_fn()
                val_hr_now  = val_metrics.get("HR@10", 0.0)
                self.val_hr.append(val_hr_now)
                val_str = (f"| Val HR@10: {val_hr_now:.4f} "
                           f"NDCG@10: {val_metrics.get('NDCG@10', 0):.4f}")
                if val_hr_now > best_hr + self.min_delta:
                    best_hr      = val_hr_now
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_ctr = 0
                else:
                    patience_ctr += 1

            if verbose:
                print(f"  Epoch {epoch:>3}/{self.n_epochs} "
                      f"| BCE: {bce:.4f} {val_str}"
                      f"| {time.time()-start:.1f}s")

            if val_eval_fn and patience_ctr >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping tại epoch {epoch} "
                      f"(HR@10 không cải thiện {self.early_stopping_patience} epoch).")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            print(f"✓ Restored best weights | Best Val HR@10: {best_hr:.4f}")
        print(f"✓ NeuMF xong sau {time.time()-start:.1f}s")

    # ─────────────────────────────────────────
    #  Inference
    # ─────────────────────────────────────────

    def score(self, users_arr: np.ndarray, items_arr: np.ndarray) -> np.ndarray:
        return self._score_batch(users_arr, items_arr)

    @torch.no_grad()
    def recommend(self, user_idx: int, n: int = 10,
                  exclude_seen: set = None) -> list:
        self.model.eval()
        all_users = torch.LongTensor([user_idx] * self.n_items).to(self.device)
        all_items = torch.arange(self.n_items, dtype=torch.long).to(self.device)
        scores    = self.model(all_users, all_items).cpu().numpy()
        if exclude_seen:
            scores[list(exclude_seen)] = -1.0
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]

    # ─────────────────────────────────────────
    #  Save / Load
    # ─────────────────────────────────────────

    def save(self, path: str):
        if not path.endswith(".pt"):
            path += ".pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "n_users":    self.n_users,
                "n_items":    self.n_items,
                "mf_dim":     self.mf_dim,
                "mlp_layers": self.mlp_layers,
                "dropout":    self.dropout,
            },
            "train_losses": self.train_losses,
            "val_hr":       self.val_hr,
        }, path)
        print(f"✓ NeuMF saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralCF":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch chưa được cài.")
        if not path.endswith(".pt"):
            path += ".pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        inst = cls(
            n_users=cfg["n_users"], n_items=cfg["n_items"],
            mf_dim=cfg["mf_dim"],   mlp_layers=cfg["mlp_layers"],
            dropout=cfg.get("dropout", 0.1),
            use_pretrain=False,     # không pre-train lại khi load
        )
        inst.model.load_state_dict(ckpt["model_state_dict"])
        inst.train_losses = ckpt.get("train_losses", [])
        inst.val_hr       = ckpt.get("val_hr", [])
        print(f"✓ NeuMF loaded ← {path}")
        return inst
