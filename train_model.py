"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  Môn: Trí Tuệ Nhân Tạo
  Kỹ thuật: Logistic Regression + MNIST
================================================
  train_model.py — Huấn luyện & lưu mô hình
================================================
"""

import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. TẢI DỮ LIỆU MNIST
# ============================================================
print("=" * 50)
print("  BƯỚC 1: TẢI DỮ LIỆU MNIST")
print("=" * 50)

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data          # (70000, 784) — mỗi ảnh 28x28 = 784 pixel
y = mnist.target.astype(int)  # nhãn 0–9

print(f"  Tổng số mẫu  : {X.shape[0]:,}")
print(f"  Số đặc trưng : {X.shape[1]} (28×28 pixel)")
print(f"  Phân phối nhãn:")
for digit in range(10):
    count = np.sum(y == digit)
    print(f"    Chữ số {digit}: {count:,} mẫu")

# ============================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ============================================================
print("\n" + "=" * 50)
print("  BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 50)

# Chuẩn hóa pixel về [0, 1]
X = X / 255.0
print(f"  Chuẩn hóa pixel: [0, 255] → [0.0, 1.0]")

# Chia tập train / test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # đảm bảo phân phối đều giữa các lớp
)

print(f"  Tập huấn luyện : {X_train.shape[0]:,} mẫu")
print(f"  Tập kiểm tra   : {X_test.shape[0]:,} mẫu")

# ============================================================
# 3. XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH
# ============================================================
print("\n" + "=" * 50)
print("  BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
print("=" * 50)
print("  Thuật toán : Logistic Regression")
print("  Solver     : SAGA (phù hợp dữ liệu lớn)")
print("  Max iter   : 1000")
# Xóa dòng hiển thị Multiclass nếu muốn code sạch hơn
print()

model = LogisticRegression(
    max_iter=1000,
    solver='saga',          # tốt cho bộ dữ liệu lớn, hỗ trợ L1/L2
    C=0.1,                  # regularization (tránh overfitting)
    tol=0.01,               # dung sai hội tụ
    n_jobs=-1,              # dùng toàn bộ CPU
    random_state=42,
    verbose=1
)

start = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start

print(f"\n  Thời gian huấn luyện: {elapsed:.1f} giây")

# ============================================================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ============================================================
print("\n" + "=" * 50)
print("  BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH")
print("=" * 50)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"  Độ chính xác trên tập test: {acc:.4f} ({acc*100:.2f}%)")
print()
print("  Báo cáo chi tiết:")
print(classification_report(y_test, y_pred,
                            target_names=[str(i) for i in range(10)]))

# ============================================================
# 5. LƯU MÔ HÌNH
# ============================================================
print("=" * 50)
print("  BƯỚC 5: LƯU MÔ HÌNH")
print("=" * 50)

joblib.dump(model, "digit_model.pkl")
print("  Mô hình đã lưu tại: digit_model.pkl")

# ============================================================
# 6. VISUALIZE MỘT SỐ MẪU DỰ ĐOÁN
# ============================================================
print("\n  Hiển thị một số mẫu dự đoán...")

fig, axes = plt.subplots(2, 10, figsize=(16, 4))
fig.suptitle("Một số mẫu dự đoán trên tập test", fontsize=13, y=1.02)

for i, ax in enumerate(axes[0]):
    idx = np.where(y_test == i)[0][0]
    img = X_test[idx].reshape(28, 28)
    pred = model.predict([X_test[idx]])[0]
    proba = model.predict_proba([X_test[idx]])[0][pred]

    ax.imshow(img, cmap='gray')
    color = 'green' if pred == y_test[idx] else 'red'
    ax.set_title(f"GT:{y_test[idx]}\nPR:{pred}", fontsize=8, color=color)
    ax.axis('off')

for i, ax in enumerate(axes[1]):
    wrong_idx = np.where((y_test != y_pred) & (y_test == i))
    if len(wrong_idx[0]) > 0:
        idx = wrong_idx[0][0]
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='Reds', alpha=0.8)
        ax.set_title(f"GT:{y_test[idx]}\nPR:{y_pred[idx]}", fontsize=8, color='red')
    else:
        ax.set_title("OK", fontsize=8, color='green')
    ax.axis('off')

axes[0][0].set_ylabel("Đúng", fontsize=9)
axes[1][0].set_ylabel("Sai", fontsize=9)

plt.tight_layout()
plt.savefig("training_samples.png", dpi=120, bbox_inches='tight')
plt.show()
print("  Biểu đồ mẫu đã lưu: training_samples.png")
print("\n  HOÀN THÀNH!")