"""
CineAI — Flask Web App
NCF: implicit feedback, score [0,1] (xác suất tương tác)
MF : explicit rating, predict [1,5]

Endpoints:
    GET /                              → Giao diện web
    GET /recommend?user_id=1&n=10&model=mf|ncf
    GET /similar?movie_id=1&n=10&model=mf|ncf
    GET /compare?user_id=1&n=5
    GET /stats
    GET /users
"""
import os, sys, time, traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(traceback.format_exc())
    return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  CineAI — Khởi động")
print("=" * 55)

from data.download_data import load_data
from utils.preprocessing import (
    encode_ids, split_data,
    leave_one_out_split, convert_to_implicit, build_negative_pool,
)
from models.matrix_factorization import MatrixFactorizationSGD
from models.neural_cf import NeuralCF

# ── 1. Load data ─────────────────────────────────────────────
ratings, movies = load_data()
df, user2idx, item2idx, idx2user, idx2item, N_USERS, N_ITEMS = encode_ids(ratings)

# ── 2. Splits ─────────────────────────────────────────────────
# MF  — explicit, random split
train_expl, temp_expl = split_data(df, test_size=0.3, random_state=42)
val_expl, test_expl   = split_data(temp_expl, test_size=0.5, random_state=42)

# NCF — implicit, leave-one-out
implicit_df                   = convert_to_implicit(df)
train_impl, val_impl, test_impl = leave_one_out_split(implicit_df)
negative_pool, interacted       = build_negative_pool(
    train_impl, N_USERS, N_ITEMS, extra_dfs=[val_impl, test_impl]
)

# Seen items per user (từ train implicit, dùng để exclude khi recommend)
seen_items = interacted   # {user_idx: set(item_idx)}

SAVED    = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVED, exist_ok=True)
MF_PATH  = os.path.join(SAVED, "mf_model")
NCF_PATH = os.path.join(SAVED, "ncf_model.pt")

# ── 3. Matrix Factorization ───────────────────────────────────
print("\n[1/2] Matrix Factorization...")
train_arr = train_expl[["user_idx", "item_idx", "rating"]].values
val_arr   = val_expl[["user_idx",   "item_idx", "rating"]].values

if os.path.exists(MF_PATH + ".npz"):
    mf_model = MatrixFactorizationSGD.load(MF_PATH)
    if getattr(mf_model, "n_items", None) != N_ITEMS or \
       getattr(mf_model, "n_users", None) != N_USERS:
        print("  ⚠ Model cũ không tương thích — train lại...")
        mf_model = MatrixFactorizationSGD(N_USERS, N_ITEMS,
                       n_factors=50, lr=0.005, reg=0.02, n_epochs=20)
        mf_model.fit(train_arr, val_arr, verbose=False)
        mf_model.save(MF_PATH)
else:
    mf_model = MatrixFactorizationSGD(N_USERS, N_ITEMS,
                   n_factors=50, lr=0.005, reg=0.02, n_epochs=20)
    mf_model.fit(train_arr, val_arr, verbose=False)
    mf_model.save(MF_PATH)
print("  ✓ MF sẵn sàng")

# ── 4. Neural CF ──────────────────────────────────────────────
print("\n[2/2] Neural CF (NeuMF — implicit)...")
ncf_model  = None
ncf_status = "not_trained"

try:
    if os.path.exists(NCF_PATH):
        ncf_model = NeuralCF.load(NCF_PATH)
        if getattr(ncf_model, "n_items", None) != N_ITEMS or \
           getattr(ncf_model, "n_users", None) != N_USERS:
            print("  ⚠ Model cũ không tương thích — train lại...")
            ncf_model = None

    if ncf_model is None:
        print("  Chưa có file — train NCF (implicit, BCE loss)...")
        ncf_model = NeuralCF(
            N_USERS, N_ITEMS,
            mf_dim=32, mlp_layers=[64, 32, 16, 8],
            dropout=0.0, lr=1e-3,
            n_epochs=20, batch_size=256, n_neg=4,
        )
        ncf_model.fit(train_impl, interacted, verbose=False)
        ncf_model.save(NCF_PATH)

    ncf_status = "ready"
    print("  ✓ NCF sẵn sàng")

except ImportError as e:
    ncf_status = "torch_not_installed"
    print(f"  ⚠ {e}")
except Exception as e:
    ncf_status = f"error: {e}"
    print(f"  ⚠ NCF lỗi: {e}")

print(f"\n🎬  CineAI — http://localhost:5000\n")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def get_model(model_type: str):
    if model_type == "ncf":
        if ncf_model is None:
            return None, f"NCF chưa sẵn sàng ({ncf_status})"
        return ncf_model, None
    return mf_model, None


def format_movie(item_idx: int, score: float, model_type: str) -> dict:
    """
    Format thông tin phim.
    - MF  : score = predicted rating [1,5]
    - NCF : score = xác suất tương tác [0,1]
    """
    movie_id = idx2item.get(item_idx, -1)
    row = movies[movies["movieId"] == movie_id]
    score_label = "predicted_rating" if model_type == "mf" else "interaction_prob"
    result = {
        "item_idx":    item_idx,
        "movie_id":    int(movie_id),
        "title":       row.iloc[0]["title"]  if not row.empty else f"Movie {movie_id}",
        "genres":      row.iloc[0]["genres"] if not row.empty else "",
        score_label:   round(float(score), 4),
    }
    return result


def get_item_vectors(model, model_type: str) -> np.ndarray:
    """
    Lấy item embedding để tính cosine similarity.
    - MF  → model.V              (n_items, n_factors)
    - NCF → GMF item embedding   (n_items, mf_dim)
    """
    if model_type == "ncf":
        # Lấy GMF item embedding weights từ NeuMF
        import torch
        with torch.no_grad():
            return model.model.gmf_item_emb.weight.cpu().numpy()
    return model.V   # MF item latent matrix


def cosine_sim_topn(V: np.ndarray, item_idx: int, n: int):
    vec   = V[item_idx]
    norms = np.linalg.norm(V, axis=1) + 1e-9
    sims  = (V @ vec) / (norms * (np.linalg.norm(vec) + 1e-9))
    sims[item_idx] = -1
    top_idx = np.argsort(sims)[::-1][:n]
    return top_idx, sims


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend")
def recommend():
    """
    GET /recommend?user_id=<int>&n=<int>&model=mf|ncf

    MF  → recommend() trả về (item_idx, predicted_rating)
    NCF → recommend() trả về (item_idx, interaction_prob)
    """
    try:
        user_id    = int(request.args.get("user_id", 1))
        n          = min(int(request.args.get("n", 10)), 50)
        model_type = request.args.get("model", "mf").lower()
    except ValueError:
        return jsonify({"error": "Tham số không hợp lệ"}), 400

    if model_type not in ("mf", "ncf"):
        return jsonify({"error": "model phải là 'mf' hoặc 'ncf'"}), 400
    if user_id not in user2idx:
        return jsonify({"error": f"Không tìm thấy user_id={user_id}"}), 404

    model, err = get_model(model_type)
    if err:
        return jsonify({"error": err}), 503

    user_idx = user2idx[user_id]
    exclude  = seen_items.get(user_idx, set())

    t0   = time.perf_counter()
    recs = model.recommend(user_idx, n=n, exclude_seen=exclude)
    ms   = round((time.perf_counter() - t0) * 1000, 2)

    score_note = (
        "Predicted rating [1-5]" if model_type == "mf"
        else "Interaction probability [0-1] — implicit NCF (He et al. 2017)"
    )

    return jsonify({
        "user_id":           user_id,
        "model":             model_type.upper(),
        "score_meaning":     score_note,
        "n_recommendations": len(recs),
        "n_seen_movies":     len(exclude),
        "inference_ms":      ms,
        "recommendations":   [format_movie(i, r, model_type) for i, r in recs],
    })


@app.route("/similar")
def similar_movies():
    """
    GET /similar?movie_id=<int>&n=<int>&model=mf|ncf

    Cosine similarity trong không gian latent:
    - MF  → item latent vector V
    - NCF → GMF item embedding
    """
    try:
        movie_id   = int(request.args.get("movie_id", 1))
        n          = min(int(request.args.get("n", 10)), 50)
        model_type = request.args.get("model", "mf").lower()
    except ValueError:
        return jsonify({"error": "Tham số không hợp lệ"}), 400

    if movie_id not in item2idx:
        return jsonify({"error": f"Không tìm thấy movie_id={movie_id}"}), 404

    model, err = get_model(model_type)
    if err:
        return jsonify({"error": err}), 503

    item_idx      = item2idx[movie_id]
    V             = get_item_vectors(model, model_type)
    top_idx, sims = cosine_sim_topn(V, item_idx, n)

    q = movies[movies["movieId"] == movie_id]
    query_title  = q.iloc[0]["title"]  if not q.empty else f"Movie {movie_id}"
    query_genres = q.iloc[0]["genres"] if not q.empty else ""

    emb_note = (
        "Item latent vector (MF)"
        if model_type == "mf"
        else "GMF item embedding (NeuMF)"
    )

    similar = []
    for idx in top_idx:
        mid = idx2item.get(int(idx), -1)
        row = movies[movies["movieId"] == mid]
        if not row.empty:
            similar.append({
                "movie_id":   int(mid),
                "title":      row.iloc[0]["title"],
                "genres":     row.iloc[0]["genres"],
                "similarity": round(float(sims[idx]), 4),
            })

    return jsonify({
        "query_movie_id":    movie_id,
        "query_title":       query_title,
        "query_genres":      query_genres,
        "model":             model_type.upper(),
        "embedding_used":    emb_note,
        "similar_movies":    similar,
    })


@app.route("/compare")
def compare():
    """GET /compare?user_id=<int>&n=<int> — so sánh MF vs NCF."""
    try:
        user_id = int(request.args.get("user_id", 1))
        n       = min(int(request.args.get("n", 5)), 20)
    except ValueError:
        return jsonify({"error": "Tham số không hợp lệ"}), 400

    if user_id not in user2idx:
        return jsonify({"error": f"Không tìm thấy user_id={user_id}"}), 404

    user_idx = user2idx[user_id]
    exclude  = seen_items.get(user_idx, set())

    t0      = time.perf_counter()
    mf_recs = mf_model.recommend(user_idx, n=n, exclude_seen=exclude)
    mf_ms   = round((time.perf_counter() - t0) * 1000, 2)

    ncf_recs, ncf_ms, ncf_err = [], None, None
    if ncf_model is not None:
        t0       = time.perf_counter()
        ncf_recs = ncf_model.recommend(user_idx, n=n, exclude_seen=exclude)
        ncf_ms   = round((time.perf_counter() - t0) * 1000, 2)
    else:
        ncf_err = f"NCF chưa sẵn sàng ({ncf_status})"

    mf_set  = {i for i, _ in mf_recs}
    ncf_set = {i for i, _ in ncf_recs}
    overlap = mf_set & ncf_set

    return jsonify({
        "user_id":     user_id,
        "n_seen":      len(exclude),
        "overlap":     len(overlap),
        "overlap_pct": round(len(overlap)/n*100, 1) if n > 0 else 0,
        "note": {
            "mf_score":  "predicted_rating [1-5]",
            "ncf_score": "interaction_probability [0-1]",
        },
        "mf": {
            "inference_ms":    mf_ms,
            "recommendations": [format_movie(i, r, "mf") for i, r in mf_recs],
        },
        "ncf": {
            "inference_ms":    ncf_ms,
            "error":           ncf_err,
            "recommendations": [format_movie(i, r, "ncf") for i, r in ncf_recs],
        },
    })


@app.route("/stats")
def stats():
    return jsonify({
        "n_users":    int(N_USERS),
        "n_items":    int(N_ITEMS),
        "n_ratings":  int(len(ratings)),
        "sparsity":   round(1 - len(ratings)/(N_USERS*N_ITEMS), 6),
        "avg_rating": round(float(ratings["rating"].mean()), 3),
        "models": {
            "mf": {
                "status":      "ready",
                "algorithm":   "Matrix Factorization (SGD + Bias + L2)",
                "feedback":    "Explicit rating [1-5]",
                "n_factors":   int(mf_model.n_factors),
                "score_range": "[1, 5]",
            },
            "ncf": {
                "status":      ncf_status,
                "algorithm":   "NeuMF (GMF + MLP) — He et al. WWW 2017",
                "feedback":    "Implicit (0/1 interaction)",
                "loss":        "Binary Cross-Entropy",
                "mf_dim":      int(ncf_model.mf_dim)          if ncf_model else None,
                "mlp_layers":  list(map(int,ncf_model.mlp_layers)) if ncf_model else None,
                "score_range": "[0, 1] (interaction probability)",
            },
        },
    })


@app.route("/users")
def list_users():
    sample = [int(u) for u in sorted(user2idx.keys())[:20]]
    return jsonify({"sample_user_ids": sample})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
