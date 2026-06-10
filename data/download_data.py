"""
Script tải và chuẩn bị dữ liệu MovieLens
"""
import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens(data_dir=DATA_DIR):
    """Tải bộ dữ liệu MovieLens 1M và giải nén vào `data/ml-1m`.

    Trả về đường dẫn đến thư mục chứa dữ liệu.
    """
    zip_path = data_dir / "ml-1m.zip"
    extract_path = data_dir / "ml-1m"

    if extract_path.exists():
        print("✓ Dữ liệu đã tồn tại, bỏ qua tải xuống.")
        return extract_path

    print("Đang tải dữ liệu MovieLens 1M...")
    response = requests.get(MOVIELENS_URL, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Đang giải nén...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    try:
        os.remove(zip_path)
    except OSError:
        pass

    print(f"✓ Tải và giải nén thành công tại: {extract_path}")
    return extract_path


def load_data(data_dir=DATA_DIR):
    """Tải dữ liệu MovieLens 1M và trả về `(ratings, movies)` DataFrame.

    - `ratings`: columns = [userId, movieId, rating, timestamp]
    - `movies`: columns = [movieId, title]
    """
    ml_dir = data_dir / "ml-1m"
    if not ml_dir.exists():
        ml_dir = download_movielens(data_dir)

    # ratings.dat uses '::' separator: userId::movieId::rating::timestamp
    ratings_path = ml_dir / "ratings.dat"
    if not ratings_path.exists():
        alt = ml_dir / "ml-1m" / "ratings.dat"
        if alt.exists():
            ratings_path = alt

    ratings = pd.read_csv(ratings_path, sep=r"::", names=["userId", "movieId", "rating", "timestamp"], engine="python")

    # movies.dat uses '::' separator: movieId::title::genres
    movies_path = ml_dir / "movies.dat"
    if not movies_path.exists():
        alt = ml_dir / "ml-1m" / "movies.dat"
        if alt.exists():
            movies_path = alt

    movies = pd.read_csv(movies_path, sep=r"::", names=["movieId", "title", "genres"], engine="python", encoding="latin-1", usecols=[0,1,2])

    print(f"✓ Ratings: {ratings.shape[0]:,} bản ghi | Movies: {movies.shape[0]:,} phim")
    return ratings, movies


if __name__ == "__main__":
    download_movielens()
