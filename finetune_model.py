"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  finetune_model.py — Huấn luyện lại số 1, 8, 9
================================================

  VẤN ĐỀ:
    Số 1 (~43%), số 8, số 9 (~38%) bị nhận dạng kém
    vì nét vẽ tay dày tạo ra phân phối pixel khác MNIST.

  GIẢI PHÁP:
    Tạo dữ liệu synthetic cho từng số bị yếu, kết hợp
    với MNIST gốc rồi retrain Logistic Regression.

  CHẠY:
    python finetune_model.py

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
MODEL_IN    = "digit_model.pkl"
MODEL_OUT   = "digit_model.pkl"
CANVAS_SIZE = 300
BG_COLOR    = "#1a1a1a"
FG_COLOR    = "white"

# Số mẫu synthetic mỗi chữ số — tăng nếu vẫn còn sai
N_PER_DIGIT = {
    1: 1000,
    8: 1000,
    9: 1000,
}
OVERSAMPLE  = 3   # nhân thêm mấy lần vào tập train

# ============================================================
# PREPROCESSING — giống hệt draw_predict.py
# ============================================================
def preprocess_arr(pil_image):
    img = np.array(pil_image.convert('L'), dtype=np.float32)
    binary = (img > 50).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad   = max(5, int(max(y_max-y_min, x_max-x_min) * 0.15))
    y_min = max(0, y_min-pad);   y_max = min(img.shape[0]-1, y_max+pad)
    x_min = max(0, x_min-pad);   x_max = min(img.shape[1]-1, x_max+pad)
    crop  = img[y_min:y_max+1, x_min:x_max+1]
    h, w  = crop.shape; size = max(h, w)
    sq    = np.zeros((size, size), dtype=np.float32)
    sq[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = crop
    p20   = Image.fromarray(sq.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    c28   = np.zeros((28, 28), dtype=np.float32)
    c28[4:24, 4:24] = np.array(p20, dtype=np.float32)
    cy, cx = center_of_mass(c28)
    c28   = nd_shift(c28, [int(round(14-cy)), int(round(14-cx))],
                     mode='constant', cval=0)
    c28   = gaussian_filter(c28, sigma=0.5)
    if c28.max() > 0:
        c28 = c28 / c28.max()
    return c28.flatten()


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_digit_1(n=1000, seed=1):
    """
    Số 1 với 3 style:
      0 — thẳng đứng + serif ngang trên (kiểu in)
      1 — thân xiên nhẹ, không serif (kiểu viết tay)
      2 — đầu chéo + thân thẳng (kiểu phổ biến nhất)
    Variation: độ nghiêng ±12°, vị trí ±25px, độ dày 20–36px
    """
    rng = np.random.RandomState(seed); samples = []
    for _ in range(n):
        img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        d   = ImageDraw.Draw(img)
        cx     = CANVAS_SIZE//2 + rng.randint(-25, 25)
        stroke = rng.randint(20, 36)
        style  = rng.randint(0, 3)

        top_y = rng.randint(40, 80)
        bot_y = rng.randint(220, 270)

        if style == 0:
            # Serif trên + thân thẳng
            sx = rng.randint(15, 30)
            d.line([cx-sx, top_y+rng.randint(15,30), cx, top_y],
                   fill=FG_COLOR, width=stroke)
            d.line([cx, top_y, cx+rng.randint(-8,8), bot_y],
                   fill=FG_COLOR, width=stroke)

        elif style == 1:
            # Thân xiên đơn giản
            lean = rng.randint(-20, 20)
            d.line([cx+lean, top_y, cx, bot_y],
                   fill=FG_COLOR, width=stroke)

        else:
            # Đầu chéo + thân nghiêng nhẹ
            ox = rng.randint(10, 25)
            d.line([cx+ox, top_y, cx+ox//2, top_y+rng.randint(20,35)],
                   fill=FG_COLOR, width=stroke)
            d.line([cx+ox//2, top_y+20, cx, bot_y],
                   fill=FG_COLOR, width=stroke)

        img = img.rotate(rng.uniform(-12, 12), fillcolor=BG_COLOR)
        r   = preprocess_arr(img)
        if r is not None:
            samples.append(r)

    return np.array(samples)


def generate_digit_8(n=1000, seed=8):
    """
    Số 8: hai vòng ellipse kích thước khác nhau.
    Variation: tỉ lệ vòng trên/dưới, độ dày, góc nghiêng ±15°
    """
    rng = np.random.RandomState(seed); samples = []
    for _ in range(n):
        img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        d   = ImageDraw.Draw(img)
        cx     = CANVAS_SIZE//2 + rng.randint(-20, 20)
        cy     = CANVAS_SIZE//2 + rng.randint(-20, 20)
        r      = rng.randint(60, 100)
        stroke = rng.randint(18, 34)
        top_w  = rng.uniform(0.50, 0.82); top_h = rng.uniform(0.80, 1.10)
        bot_w  = rng.uniform(0.60, 0.90); bot_h = rng.uniform(0.80, 1.10)
        ov     = rng.uniform(0.0, 0.12)

        d.ellipse([cx-r*top_w, cy-r*top_h, cx+r*top_w, cy+r*ov],
                  outline=FG_COLOR, width=stroke)
        d.ellipse([cx-r*bot_w, cy-r*ov, cx+r*bot_w, cy+r*bot_h],
                  outline=FG_COLOR, width=stroke)

        img = img.rotate(rng.uniform(-15, 15), fillcolor=BG_COLOR)
        r2  = preprocess_arr(img)
        if r2 is not None:
            samples.append(r2)

    return np.array(samples)


def generate_digit_9(n=1000, seed=9):
    """
    Số 9: vòng tròn trên + đuôi xuống.
    Style đuôi:
      0 — thẳng đứng
      1 — cong sang trái (kiểu chữ đẹp)
      2 — xiên nhẹ sang phải
    Variation: kích thước vòng, vị trí đuôi, góc nghiêng ±15°
    """
    rng = np.random.RandomState(seed); samples = []
    for _ in range(n):
        img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        d   = ImageDraw.Draw(img)
        cx      = CANVAS_SIZE//2 + rng.randint(-20, 20)
        cy_top  = CANVAS_SIZE//2 - rng.randint(30, 60)
        r       = rng.randint(55, 90)
        stroke  = rng.randint(20, 34)
        top_w   = rng.uniform(0.70, 1.05)
        top_h   = rng.uniform(0.80, 1.10)

        # Vòng tròn
        d.ellipse([cx-r*top_w, cy_top-r*top_h,
                   cx+r*top_w, cy_top+r*top_h],
                  outline=FG_COLOR, width=stroke)

        # Đuôi
        tail_x     = cx + r*top_w*rng.uniform(0.65, 0.95)
        tail_top_y = cy_top + rng.randint(-15, 25)
        tail_bot_y = cy_top + r*top_h + rng.randint(55, 115)
        tail_bot_x = cx + rng.randint(-35, 35)
        style      = rng.randint(0, 3)

        if style == 0:
            d.line([int(tail_x), int(tail_top_y),
                    int(tail_bot_x), int(tail_bot_y)],
                   fill=FG_COLOR, width=stroke)
        elif style == 1:
            pts = []
            for t in np.linspace(0, 1, 20):
                x = tail_x + (tail_bot_x - tail_x)*t - 25*math.sin(math.pi*t)
                y = tail_top_y + (tail_bot_y - tail_top_y)*t
                pts.append((x, y))
            for i in range(len(pts)-1):
                d.line([pts[i], pts[i+1]], fill=FG_COLOR, width=stroke)
        else:
            lean_x = tail_bot_x + rng.randint(10, 30)
            d.line([int(tail_x), int(tail_top_y),
                    int(lean_x), int(tail_bot_y)],
                   fill=FG_COLOR, width=stroke)

        img = img.rotate(rng.uniform(-15, 15), fillcolor=BG_COLOR)
        r2  = preprocess_arr(img)
        if r2 is not None:
            samples.append(r2)

    return np.array(samples)


GENERATORS = {
    1: generate_digit_1,
    8: generate_digit_8,
    9: generate_digit_9,
}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 58)
    print("  FINETUNE SỐ 1 + 8 + 9 — Nhận dạng chữ số viết tay")
    print("=" * 58)

    # ── 1. Tạo synthetic data ────────────────────────────────
    print("\n[1/5] Tạo dữ liệu synthetic cho số 1, 8, 9...")
    synth_X, synth_y = [], []

    for digit, gen_fn in GENERATORS.items():
        n = N_PER_DIGIT[digit]
        print(f"  Đang tạo {n} mẫu số {digit}...", end=" ", flush=True)
        X = gen_fn(n)
        print(f"✅  {len(X)} mẫu")
        synth_X.append(X)
        synth_y.append(np.full(len(X), digit))

    synth_X = np.vstack(synth_X)
    synth_y = np.concatenate(synth_y)
    print(f"\n  Tổng synthetic: {len(synth_X):,} mẫu "
          f"(~{len(synth_X)//len(GENERATORS)} / chữ số)")

    # ── 2. Tải MNIST ─────────────────────────────────────────
    print("\n[2/5] Tải MNIST dataset...")
    try:
        mnist   = fetch_openml('mnist_784', version=1,
                               as_frame=False, cache=True)
        X_mnist = mnist.data / 255.0
        y_mnist = mnist.target.astype(int)
        print(f"  ✅  MNIST: {len(X_mnist):,} mẫu")
    except Exception as e:
        print(f"  ⚠️   Không tải được MNIST: {e}")
        print("  → Dùng synthetic only (kết quả có thể thấp hơn)")
        X_mnist = synth_X; y_mnist = synth_y

    # ── 3. Kết hợp dữ liệu ───────────────────────────────────
    print(f"\n[3/5] Kết hợp MNIST + synthetic (oversample ×{OVERSAMPLE})...")

    parts_X = [X_mnist] + [synth_X] * OVERSAMPLE
    parts_y = [y_mnist] + [synth_y] * OVERSAMPLE
    X_all   = np.vstack(parts_X)
    y_all   = np.concatenate(parts_y)

    # Shuffle
    idx     = np.random.RandomState(42).permutation(len(X_all))
    X_all   = X_all[idx]; y_all = y_all[idx]

    print(f"  Tổng mẫu: {len(X_all):,}")
    for d in range(10):
        mark = " ← augmented" if d in GENERATORS else ""
        print(f"    Số {d}: {np.sum(y_all==d):,} mẫu{mark}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
    print(f"\n  Train: {len(X_tr):,} | Test: {len(X_te):,}")

    # ── 4. Retrain ───────────────────────────────────────────
    print("\n[4/5] Huấn luyện lại Logistic Regression...")
    print("  solver=saga, max_iter=500, C=0.1 — có thể mất vài phút\n")

    clf = LogisticRegression(
        max_iter=500, solver='saga', C=0.1,
        tol=0.01, n_jobs=-1, random_state=42, verbose=1)

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    print(f"\n  ✅  Xong trong {time.time()-t0:.1f}s")

    # ── 5. Đánh giá ──────────────────────────────────────────
    print("\n[5/5] Đánh giá mô hình mới...")
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    print(f"\n  Độ chính xác tổng thể: {acc*100:.2f}%\n")
    print(classification_report(y_te, y_pred,
                                target_names=[str(i) for i in range(10)]))

    print("  🎯 Accuracy riêng các số được cải thiện:")
    for d in GENERATORS:
        mask = y_te == d
        if mask.sum() > 0:
            a = accuracy_score(y_te[mask], y_pred[mask])
            print(f"    Số {d}: {a*100:.2f}%")

    # Lưu
    joblib.dump(clf, MODEL_OUT)
    print(f"\n  ✅  Mô hình đã lưu: {MODEL_OUT}")
    print("  Chạy lại draw_predict.py để kiểm tra!")
    print("=" * 58)