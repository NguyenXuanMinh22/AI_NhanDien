"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  finetune_digit8.py — Huấn luyện lại cho số 8
================================================

  VẤN ĐỀ:
    Model nhận dạng số 8 vẽ tay thành số khác (1, 4, v.v.)
    vì nét vẽ dày trên canvas tạo ra ảnh 28×28 rất khác
    so với số 8 trong tập MNIST (nét mảnh, anti-aliased).

  GIẢI PHÁP:
    1. Tạo 1000 mẫu số 8 synthetic: vẽ bằng PIL với nhiều
       biến thể (kích thước, độ dày nét, tỉ lệ, góc nghiêng)
       → giả lập cách người dùng vẽ trên canvas
    2. Áp dụng ĐÚNG preprocessing (center-of-mass, blur)
       giống hệt draw_predict.py
    3. Kết hợp với MNIST gốc (60k mẫu) + weighted sampling
       để model nhớ tất cả các số, không quên số cũ
    4. Retrain Logistic Regression và lưu đè digit_model.pkl

  CHẠY:
    python finetune_digit8.py

  YÊU CẦU:
    pip install scipy scikit-learn pillow numpy joblib
================================================
"""

import numpy as np
import joblib
import time
import math
from PIL import Image, ImageDraw

from scipy.ndimage import gaussian_filter, center_of_mass, shift as nd_shift
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_IN    = "digit_model.pkl"   # model gốc cần load
MODEL_OUT   = "digit_model.pkl"   # ghi đè (hoặc đổi tên để giữ bản gốc)
N_SYNTH     = 1000                # số mẫu 8 synthetic tạo ra
CANVAS_SIZE = 300
BG_COLOR    = "#1a1a1a"
FG_COLOR    = "white"

# ============================================================
# HÀM PREPROCESSING — giống hệt draw_predict.py
# ============================================================
def preprocess_array(pil_image):
    """
    Áp dụng đúng chuẩn MNIST: center-of-mass + blur.
    PHẢI giống hệt hàm preprocess() trong draw_predict.py.
    """
    img = np.array(pil_image.convert('L'), dtype=np.float32)
    binary = (img > 50).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad   = max(5, int(max(y_max - y_min, x_max - x_min) * 0.15))
    y_min = max(0, y_min - pad);   y_max = min(img.shape[0]-1, y_max + pad)
    x_min = max(0, x_min - pad);   x_max = min(img.shape[1]-1, x_max + pad)
    crop  = img[y_min:y_max+1, x_min:x_max+1]

    h, w = crop.shape; size = max(h, w)
    square = np.zeros((size, size), dtype=np.float32)
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = crop

    pil_20   = Image.fromarray(square.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[4:24, 4:24] = np.array(pil_20, dtype=np.float32)

    cy, cx   = center_of_mass(canvas28)
    dy, dx   = int(round(14 - cy)), int(round(14 - cx))
    canvas28 = nd_shift(canvas28, [dy, dx], mode='constant', cval=0)
    canvas28 = gaussian_filter(canvas28, sigma=0.5)

    if canvas28.max() > 0:
        canvas28 = canvas28 / canvas28.max()

    return canvas28.flatten()


# ============================================================
# TẠO MẪU SỐ 8 SYNTHETIC
# ============================================================
def generate_digit8_samples(n=1000, seed=42):
    """
    Tạo n mẫu số 8 với nhiều biến thể để model học đa dạng:
    - Kích thước (r: 60–100px)
    - Độ dày nét (stroke: 18–34px)
    - Tỉ lệ vòng trên/dưới khác nhau
    - Độ lệch tâm nhỏ (±20px)
    - Góc nghiêng (±15°)
    Tất cả đều đi qua preprocess_array() → đảm bảo
    distribution khớp với ảnh canvas thực tế.
    """
    rng     = np.random.RandomState(seed)
    samples = []
    failed  = 0

    print(f"  Tạo {n} mẫu số 8 synthetic...")

    for i in range(n):
        img  = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        draw = ImageDraw.Draw(img)

        cx     = CANVAS_SIZE//2 + rng.randint(-20, 20)
        cy     = CANVAS_SIZE//2 + rng.randint(-20, 20)
        r      = rng.randint(60, 100)
        stroke = rng.randint(18, 34)

        # Tỉ lệ ngẫu nhiên cho vòng trên và dưới
        top_w   = rng.uniform(0.50, 0.82)
        top_h   = rng.uniform(0.80, 1.10)
        bot_w   = rng.uniform(0.60, 0.90)
        bot_h   = rng.uniform(0.80, 1.10)
        overlap = rng.uniform(0.0,  0.12)

        # Vẽ vòng trên
        draw.ellipse([
            cx - r*top_w, cy - r*top_h,
            cx + r*top_w, cy + r*overlap
        ], outline=FG_COLOR, width=stroke)

        # Vẽ vòng dưới
        draw.ellipse([
            cx - r*bot_w, cy - r*overlap,
            cx + r*bot_w, cy + r*bot_h
        ], outline=FG_COLOR, width=stroke)

        # Xoay ngẫu nhiên
        angle = rng.uniform(-15, 15)
        img   = img.rotate(angle, fillcolor=BG_COLOR)

        result = preprocess_array(img)
        if result is not None:
            samples.append(result)
        else:
            failed += 1

    print(f"  ✅ Tạo được {len(samples)} mẫu ({failed} thất bại)")
    return np.array(samples)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  FINETUNE SỐ 8 — Nhận dạng chữ số viết tay")
    print("=" * 55)

    # ── Bước 1: Tạo synthetic data cho số 8 ─────────────────
    print("\n[1/5] Tạo dữ liệu số 8 synthetic...")
    X8_synth = generate_digit8_samples(N_SYNTH)
    y8_synth = np.full(len(X8_synth), 8)
    print(f"  Synthetic 8: {X8_synth.shape} | mean={X8_synth.mean():.4f}")

    # ── Bước 2: Tải MNIST ────────────────────────────────────
    print("\n[2/5] Tải MNIST dataset...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
        X_raw = mnist.data / 255.0
        y_raw = mnist.target.astype(int)
        print(f"  MNIST loaded: {X_raw.shape[0]:,} mẫu")
    except Exception as e:
        print(f"  ⚠️  Không tải được MNIST: {e}")
        print("  Tiếp tục với synthetic data only...")
        X_raw, y_raw = X8_synth, y8_synth

    # ── Bước 3: Kết hợp dữ liệu ─────────────────────────────
    print("\n[3/5] Kết hợp MNIST + synthetic data...")

    # Lấy toàn bộ MNIST, thêm synthetic 8
    # Để cân bằng: oversample synthetic 8 thêm 2× nữa
    X_combined = np.vstack([
        X_raw,
        X8_synth,
        X8_synth,   # oversample 2×
        X8_synth,   # oversample 3×
    ])
    y_combined = np.concatenate([
        y_raw,
        y8_synth,
        y8_synth,
        y8_synth,
    ])

    # Shuffle
    rng_idx = np.random.RandomState(123)
    idx     = rng_idx.permutation(len(X_combined))
    X_combined = X_combined[idx]
    y_combined = y_combined[idx]

    print(f"  Tổng mẫu: {len(X_combined):,}")
    for d in range(10):
        print(f"    Số {d}: {np.sum(y_combined == d):,} mẫu")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined,
        test_size=0.1, random_state=42, stratify=y_combined)

    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Bước 4: Retrain ──────────────────────────────────────
    print("\n[4/5] Huấn luyện lại Logistic Regression...")
    print("  (Có thể mất vài phút — solver=saga, max_iter=500)")

    model = LogisticRegression(
        max_iter=500,
        solver='saga',
        C=0.1,
        tol=0.01,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  Huấn luyện xong trong {elapsed:.1f}s")

    # ── Bước 5: Đánh giá ─────────────────────────────────────
    print("\n[5/5] Đánh giá mô hình...")

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Độ chính xác tổng thể: {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=[str(i) for i in range(10)]))

    # Hiển thị riêng accuracy cho số 8
    mask8     = y_test == 8
    acc8      = accuracy_score(y_test[mask8], y_pred[mask8])
    print(f"  🎯 Độ chính xác riêng số 8: {acc8:.4f} ({acc8*100:.2f}%)")

    # Lưu model
    joblib.dump(model, MODEL_OUT)
    print(f"\n  ✅ Mô hình đã lưu: {MODEL_OUT}")
    print("  Hãy chạy lại draw_predict.py để kiểm tra!")
    print("=" * 55)