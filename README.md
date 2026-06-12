# CineAI — Deep Learning Recommender System

Hệ thống gợi ý phim theo đúng paper **Neural Collaborative Filtering (He et al., WWW 2017)**,
xây dựng với PyTorch trên bộ dữ liệu **MovieLens 1M**.

---

## Kiến trúc

### NeuMF — Neural Matrix Factorization (đúng paper)

```
User Embedding (GMF)      User Embedding (MLP)
Item Embedding (GMF)      Item Embedding (MLP)
      ⊙ (Hadamard)              Concat
     GMF Output           MLP [64→32→16→8]  ← ReLU activation
           ↘                   ↙
             Concat → Linear → Sigmoid → [0,1]
                   (xác suất tương tác)
```

| Thành phần | Chi tiết |
|---|---|
| **Feedback type** | Implicit (0/1 — quan sát hay không quan sát) |
| **Loss function** | Binary Cross-Entropy (log loss) — đúng paper |
| **Negative sampling** | 4 negatives/positive mỗi epoch (dynamic) — đúng paper |
| **Evaluation** | Leave-one-out: 1 test item + 99 sampled negatives → HR@10, NDCG@10 |
| **Embedding** | GMF và MLP dùng embedding riêng biệt — không share |
| **MLP activation** | ReLU (paper: ReLU > tanh > sigmoid) |
| **Output** | Sigmoid → xác suất [0, 1] |

### MF-SGD (Baseline — explicit rating)

Matrix Factorization truyền thống, dự đoán rating 1–5, loss MSE.
Dùng để so sánh với NeuMF trên cùng dataset.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

**Yêu cầu:** Python ≥ 3.8, PyTorch ≥ 1.9.
Dữ liệu MovieLens 1M tự động tải khi chạy lần đầu.

---

## Cách chạy

### Train đầy đủ (NCF + MF + toàn bộ output)
```bash
python train.py --model all --epochs 20 --save
```

### Chỉ NeuMF (implicit, đúng paper)
```bash
python train.py --model ncf --epochs 20 --save
```

### Chỉ MF (explicit baseline)
```bash
python train.py --model mf --epochs 20 --save
```

Sau khi chạy, `saved_models/` sinh ra:
```
saved_models/
├── ncf_model.pt           ← NeuMF weights
├── mf_model.npz           ← MF weights
├── results.json           ← HR@10, NDCG@10 (NCF) | RMSE, MAE (MF)
├── training_curves.png    ← BCE loss + Val HR@10 theo epoch
└── model_comparison.png   ← so sánh HR@10, NDCG@10
```

### Unit tests
```bash
pytest tests/ -v
```

### Web App
```bash
python app.py
# → http://localhost:5000
```

---

## Tham số CLI

| Tham số | Default | Mô tả |
|---|---|---|
| `--model` | `all` | `ncf` / `mf` / `all` |
| `--epochs` | `20` | Số epoch |
| `--seed` | `42` | Random seed |
| `--topk` | `10` | K cho HR@K, NDCG@K |
| `--topk_eval` | `1000` | Số users dùng để evaluate |
| `--save` | flag | Lưu model weights |
| **NCF** | | |
| `--mf_dim` | `32` | GMF embedding dim (paper: 8/16/32/64) |
| `--mlp_layers` | `64 32 16 8` | MLP tower sizes |
| `--ncf_lr` | `0.001` | Learning rate (Adam) |
| `--batch_size` | `256` | Batch size |
| `--n_neg` | `4` | Negative samples/positive (paper: 4) |
| `--dropout` | `0.0` | Dropout (paper không dùng) |
| `--early_stop` | `5` | Early stopping patience (theo Val HR@10) |
| **MF** | | |
| `--lr` | `0.005` | Learning rate (SGD) |
| `--reg` | `0.02` | L2 regularization |
| `--mf_factors` | `50` | Số latent factors |

---

## Điểm khác biệt so với MF thông thường (theo paper)

| | Matrix Factorization | NeuMF (paper) |
|---|---|---|
| Feedback | Explicit rating 1–5 | Implicit 0/1 |
| Interaction function | Inner product (tuyến tính) | GMF ⊙ + MLP (phi tuyến) |
| Loss | MSE | Binary Cross-Entropy |
| Negative | Không | Sample 4 neg/pos mỗi epoch |
| Evaluation | RMSE, MAE | HR@K, NDCG@K |
| Split | Random 70/15/15 | Leave-one-out |

---

## Cấu trúc project

```
recommender_system/
├── data/
│   ├── ml-1m/                   # Dataset (tự động tải)
│   └── download_data.py
├── models/
│   ├── matrix_factorization.py  # MF-SGD (explicit baseline)
│   └── neural_cf.py             # NeuMF theo đúng paper He et al. 2017
├── utils/
│   ├── preprocessing.py         # encode IDs, leave-one-out split,
│   │                            # negative pool (exclude val+test)
│   └── metrics.py               # HR@K, NDCG@K (implicit) | RMSE, MAE (explicit)
├── notebooks/
│   └── demo_recommender.py      # Demo pipeline
├── tests/
│   └── test_models.py           # 27 unit tests (pytest)
├── saved_models/                # Trống — sinh ra khi chạy train.py
├── templates/index.html
├── app.py                       # Flask web app
├── train.py                     # Script train
├── BaoCao_RecommenderSystem_DeepLearning.docx
└── requirements.txt
```

---

## Tài liệu tham khảo

- **He, X., Liao, L., Zhang, H., et al. (2017).** *Neural Collaborative Filtering.* WWW 2017.
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* Computer.
- Harper, F.M., & Konstan, J.A. (2015). *The MovieLens Datasets.* ACM TIIS.
