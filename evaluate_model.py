"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  evaluate_model.py — Đánh giá & trực quan hóa
================================================
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# ============================================================
# 1. TẢI MÔ HÌNH & DỮ LIỆU
# ============================================================
print("Đang tải mô hình và dữ liệu...")

model = joblib.load("digit_model.pkl")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data / 255.0
y = mnist.target.astype(int)

# Dùng 10,000 mẫu để đánh giá nhanh
X_eval = X[:10000]
y_eval = y[:10000]

y_pred = model.predict(X_eval)
y_proba = model.predict_proba(X_eval)

acc = accuracy_score(y_eval, y_pred)
print(f"Độ chính xác: {acc:.4f} ({acc*100:.2f}%)\n")
print(classification_report(y_eval, y_pred))

# ============================================================
# 2. CONFUSION MATRIX
# ============================================================
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
cm = confusion_matrix(y_eval, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='#eee')
ax1.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=13, pad=10)
ax1.set_xlabel('Dự đoán', fontsize=11)
ax1.set_ylabel('Thực tế', fontsize=11)

# ============================================================
# 3. ĐỘ CHÍNH XÁC TỪNG CHỮ SỐ
# ============================================================
ax2 = fig.add_subplot(gs[0, 2])
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['#2196F3' if a >= 0.95 else '#FF9800' if a >= 0.9 else '#F44336'
          for a in per_class_acc]

bars = ax2.barh(range(10), per_class_acc, color=colors)
ax2.set_yticks(range(10))
ax2.set_yticklabels([f'Số {i}' for i in range(10)])
ax2.set_xlabel('Độ chính xác', fontsize=10)
ax2.set_title('Độ chính xác từng chữ số', fontsize=11)
ax2.set_xlim(0.85, 1.0)

for bar, val in zip(bars, per_class_acc):
    ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left', fontsize=8)

# ============================================================
# 4. MỘT SỐ MẪU SAI
# ============================================================
ax3 = fig.add_subplot(gs[1, :])
wrong_mask = y_eval != y_pred
wrong_idx = np.where(wrong_mask)[0][:20]

if len(wrong_idx) > 0:
    n = min(20, len(wrong_idx))
    for i, idx in enumerate(wrong_idx[:n]):
        sub = ax3.inset_axes([i/n, 0, 1/n - 0.005, 0.9])
        sub.imshow(X_eval[idx].reshape(28, 28), cmap='gray')
        sub.set_title(f'GT:{y_eval[idx]}\nPR:{y_pred[idx]}',
                      fontsize=7, color='red')
        sub.axis('off')
    ax3.axis('off')
    ax3.set_title('Một số mẫu dự đoán sai', fontsize=11, y=0.95)

fig.suptitle(f'Đánh giá mô hình Logistic Regression — Độ chính xác: {acc:.2%}',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('evaluation_results.png', dpi=130, bbox_inches='tight')
plt.show()
print("Đã lưu biểu đồ: evaluation_results.png")