"""
Train & Evaluate Recommender System theo paper NCF (He et al. 2017).

Sử dụng:
    python train.py --model ncf         # NeuMF với implicit feedback (đúng paper)
    python train.py --model mf          # MF baseline với explicit rating
    python train.py --model all         # Cả hai + so sánh

Output (saved_models/):
    ncf_model.pt          ← NeuMF weights
    mf_model.npz          ← MF weights
    results.json          ← bảng số liệu
    training_curves.png   ← BCE loss (NCF) / RMSE (MF) theo epoch
    model_comparison.png  ← so sánh HR@10, NDCG@10
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.download_data import load_data
from utils.preprocessing import (
    encode_ids, leave_one_out_split, convert_to_implicit,
    build_negative_pool, get_statistics, split_data,
)
from utils.metrics import (
    rmse, mae, evaluate_implicit, print_metrics,
)
from models.matrix_factorization import MatrixFactorizationSGD
from models.neural_cf import NeuralCF

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


# ── Helpers ────────────────────────────────────────────────────────────────────

def savefig(fname: str):
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved_models/{fname}")


# ── Training curves ────────────────────────────────────────────────────────────

def save_training_curves_ncf(train_losses, val_hr):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("NeuMF Training Curves (MovieLens 1M)", fontsize=13, fontweight="bold")

    ep = range(1, len(train_losses)+1)
    axes[0].plot(ep, train_losses, "b-o", ms=4, label="Train BCE Loss")
    axes[0].set_title("BCE Loss (per epoch)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss"); axes[0].legend()

    if val_hr:
        axes[1].plot(range(1, len(val_hr)+1), val_hr, "r-s", ms=4, label="Val HR@10")
        axes[1].set_title("Validation HR@10 (per epoch)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("HR@10"); axes[1].legend()
    else:
        axes[1].set_title("No validation data")

    plt.tight_layout()
    savefig("training_curves.png")


def save_training_curves_both(mf_rmse_train, mf_rmse_val, ncf_bce, ncf_val_hr):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves — MF vs NeuMF (MovieLens 1M)", fontsize=13, fontweight="bold")

    ep_mf  = range(1, len(mf_rmse_train)+1)
    axes[0].plot(ep_mf, mf_rmse_train, "b-o", ms=4, label="Train RMSE")
    axes[0].plot(ep_mf, mf_rmse_val,   "r-s", ms=4, label="Val RMSE")
    axes[0].set_title("MF — RMSE"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE"); axes[0].legend()

    ep_ncf = range(1, len(ncf_bce)+1)
    axes[1].plot(ep_ncf, ncf_bce, "b-o", ms=4, label="Train BCE Loss")
    axes[1].set_title("NeuMF — BCE Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE Loss"); axes[1].legend()

    if ncf_val_hr:
        axes[2].plot(range(1, len(ncf_val_hr)+1), ncf_val_hr, "r-s", ms=4, label="Val HR@10")
        axes[2].set_title("NeuMF — Val HR@10"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("HR@10"); axes[2].legend()
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    savefig("training_curves.png")


# ── Comparison chart ───────────────────────────────────────────────────────────

def save_comparison_chart(results, topk):
    """Bar chart so sánh các metric giữa MF và NCF."""
    # Chỉ lấy metric chung
    common_metrics = [f"HR@{topk}", f"NDCG@{topk}"]
    available = [m for m in common_metrics
                 if all(m in r for r in results.values())]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(available)); width = 0.35
    colors = ["#4C72B0", "#DD8452"]
    for i, (model, color) in enumerate(zip(results.keys(), colors)):
        vals = [results[model][m] for m in available]
        bars = ax.bar(x + i*width, vals, width*0.9, label=model, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width/2); ax.set_xticklabels(available)
    ax.set_title("Model Comparison — Implicit Feedback (MovieLens 1M)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    savefig("model_comparison.png")


# ── Train NCF ──────────────────────────────────────────────────────────────────

def train_ncf(train_df, val_df, test_df, n_users, n_items,
              interacted, negative_pool, args):
    print("\n" + "="*55)
    print("  NEURAL COLLABORATIVE FILTERING (NeuMF)")
    print("  Implicit feedback | BCE loss | Negative sampling")
    print("="*55)

    ncf = NeuralCF(
        n_users=n_users, n_items=n_items,
        mf_dim=args.mf_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        lr=args.ncf_lr,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_neg=args.n_neg,
        seed=args.seed,
        early_stopping_patience=args.early_stop,
        use_pretrain=args.use_pretrain,
        pretrain_epochs=args.pretrain_epochs,
        alpha=args.alpha,
    )

    # Validation function: evaluate HR@10, NDCG@10 trên val set
    def val_eval():
        return evaluate_implicit(
            model_score_fn=ncf.score,
            test_df=val_df,
            negative_pool=negative_pool,
            n_neg=99,
            k=args.topk,
            seed=args.seed,
        )

    ncf.fit(train_df, interacted, val_eval_fn=val_eval, verbose=True)

    # Đánh giá trên test set (đúng chuẩn paper: 1 positive + 99 negatives)
    print(f"\n  Đánh giá trên Test set ({args.topk_eval} users)...")
    test_subset = test_df.sample(
        min(args.topk_eval, len(test_df)), random_state=args.seed
    )
    test_metrics = evaluate_implicit(
        model_score_fn=ncf.score,
        test_df=test_subset,
        negative_pool=negative_pool,
        n_neg=99,
        k=args.topk,
        seed=args.seed,
    )
    print_metrics(test_metrics)

    if args.save:
        ncf.save(os.path.join(OUTDIR, "ncf_model.pt"))

    return ncf, test_metrics


# ── Train MF ───────────────────────────────────────────────────────────────────

def train_mf(train_df, val_df, test_df, n_users, n_items, args):
    print("\n" + "="*55)
    print("  MATRIX FACTORIZATION (SGD) — Explicit baseline")
    print("="*55)

    train_arr = train_df[["user_idx", "item_idx", "rating"]].values
    val_arr   = val_df[["user_idx", "item_idx", "rating"]].values
    test_arr  = test_df[["user_idx", "item_idx", "rating"]].values

    mf = MatrixFactorizationSGD(
        n_users=n_users, n_items=n_items,
        n_factors=args.mf_factors, lr=args.lr,
        reg=args.reg, n_epochs=args.epochs,
        random_state=args.seed,
    )
    mf.fit(train_arr, val_arr)

    y_true = test_arr[:, 2]
    y_pred = mf.predict_batch(test_arr[:, :2])
    result = {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE":  round(mae(y_true, y_pred), 4),
    }
    print_metrics(result)

    if args.save:
        mf.save(os.path.join(OUTDIR, "mf_model"))
        print("  ✓ saved_models/mf_model.npz")

    return mf, result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Recommender System — NCF (implicit) + MF (explicit)"
    )
    parser.add_argument("--model",       choices=["ncf","mf","all"], default="all")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--topk",        type=int,   default=10,
                        help="K cho HR@K và NDCG@K")
    parser.add_argument("--topk_eval",   type=int,   default=1000,
                        help="Số users dùng để evaluate (paper: tất cả, dùng ít hơn cho nhanh)")
    parser.add_argument("--save",        action="store_true")
    # NCF
    parser.add_argument("--mf_dim",      type=int,   default=32,
                        help="GMF embedding dim (paper: 8/16/32/64)")
    parser.add_argument("--mlp_layers",  type=int,   nargs="+", default=[64,32,16,8],
                        help="MLP tower sizes, ví dụ: --mlp_layers 64 32 16 8")
    parser.add_argument("--ncf_lr",      type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--dropout",     type=float, default=0.1,
                        help="Dropout trong MLP (0.1 giúp giảm overfitting)")
    parser.add_argument("--use_pretrain", action="store_true", default=True,
                        help="Pre-train GMF+MLP trước khi train NeuMF (Section 3.4.1)")
    parser.add_argument("--no_pretrain",  dest="use_pretrain", action="store_false",
                        help="Bỏ qua pre-training")
    parser.add_argument("--pretrain_epochs", type=int, default=10,
                        help="Số epochs cho GMF/MLP pre-training")
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="Trade-off GMF vs MLP khi khởi tạo NeuMF (paper: 0.5)")
    parser.add_argument("--n_neg",       type=int,   default=4,
                        help="Negative samples per positive (paper: 4)")
    parser.add_argument("--weight_decay",type=float, default=0.0)
    parser.add_argument("--early_stop",  type=int,   default=5)
    # MF
    parser.add_argument("--lr",          type=float, default=0.005)
    parser.add_argument("--reg",         type=float, default=0.02)
    parser.add_argument("--mf_factors",  type=int,   default=50)

    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(args.seed)

    # Cảnh báo overwrite
    existing = [f for f in os.listdir(OUTDIR)
                if f != ".gitkeep" and not f.startswith(".")]
    if existing:
        print("\n" + "="*50)
        print("  ⚠  CẢNH BÁO: saved_models/ đã có file cũ")
        print("="*50)
        for f in sorted(existing): print(f"  - {f}")
        print("  → Các file trên sẽ bị GHI ĐÈ khi train xong.")
        print("="*50)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n" + "="*55)
    print("  BƯỚC 1 — Tải dữ liệu")
    print("="*55)
    ratings, movies = load_data()
    get_statistics(ratings, movies)

    # ── 2. Encode ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  BƯỚC 2 — Tiền xử lý")
    print("="*55)
    df, user2idx, item2idx, idx2user, idx2item, n_users, n_items = encode_ids(ratings)

    # ── 3. Split ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  BƯỚC 4 — Chia tập dữ liệu")
    print("="*55)

    # Implicit: leave-one-out (cho NCF)
    implicit_df              = convert_to_implicit(df)
    train_impl, val_impl, test_impl = leave_one_out_split(implicit_df)
    negative_pool, interacted = build_negative_pool(train_impl, n_users, n_items,
                                                        extra_dfs=[val_impl, test_impl])

    # Explicit: random split (cho MF baseline)
    train_expl, temp_expl = split_data(df, test_size=0.3, random_state=args.seed)
    val_expl, test_expl   = split_data(temp_expl, test_size=0.5, random_state=args.seed)

    # ── 5. Train ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  BƯỚC 5 — Huấn luyện")
    print("="*55)

    all_results = {}
    ncf_model = mf_model = None

    if args.model in ("ncf", "all"):
        ncf_model, ncf_res = train_ncf(
            train_impl, val_impl, test_impl,
            n_users, n_items, interacted, negative_pool, args
        )
        all_results["NeuMF (Implicit)"] = ncf_res

    if args.model in ("mf", "all"):
        mf_model, mf_res = train_mf(
            train_expl, val_expl, test_expl,
            n_users, n_items, args
        )
        all_results["Matrix Factorization (Explicit)"] = mf_res

    # ── 6. Lưu kết quả ────────────────────────────────────────
    print("\n" + "="*55)
    print("  BƯỚC 6 — Lưu kết quả & biểu đồ")
    print("="*55)

    with open(os.path.join(OUTDIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("  ✓ saved_models/results.json")

    # Training curves
    if args.model == "all" and ncf_model and mf_model:
        save_training_curves_both(
            mf_model.train_losses, mf_model.val_losses,
            ncf_model.train_losses, ncf_model.val_hr,
        )
    elif args.model == "ncf" and ncf_model:
        save_training_curves_ncf(ncf_model.train_losses, ncf_model.val_hr)
    elif args.model == "mf" and mf_model:
        fig, ax = plt.subplots(figsize=(9,5))
        ep = range(1, len(mf_model.train_losses)+1)
        ax.plot(ep, mf_model.train_losses, "b-o", ms=4, label="Train RMSE")
        ax.plot(ep, mf_model.val_losses,   "r-s", ms=4, label="Val RMSE")
        ax.set_title("MF Training Curve"); ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE"); ax.legend()
        savefig("training_curves.png")

    if len(all_results) > 1:
        save_comparison_chart(all_results, args.topk)



    # ── 7. Tổng kết ───────────────────────────────────────────
    print("\n" + "="*55)
    print("  KẾT QUẢ CUỐI CÙNG")
    print("="*55)
    for model_name, res in all_results.items():
        print(f"\n  {model_name}:")
        for k, v in res.items():
            if isinstance(v, float): print(f"    {k:<25}: {v:.4f}")
            elif isinstance(v, int): print(f"    {k:<25}: {v}")

    print("\n" + "="*55)
    print("  FILES SINH RA trong saved_models/")
    print("="*55)
    for fname in sorted(os.listdir(OUTDIR)):
        if fname in (".gitkeep",): continue
        size = os.path.getsize(os.path.join(OUTDIR, fname))
        print(f"  {fname:<38} {size/1024:>7.1f} KB")


if __name__ == "__main__":
    main()
