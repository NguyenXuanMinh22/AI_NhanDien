"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  draw_predict.py — v3 (FINAL FIX)
================================================

  🔍 NGUYÊN NHÂN GỐC RỄ GÂY NHẬN DẠNG SAI:
  ─────────────────────────────────────────────
  LỖI 1 — Đảo màu sai (đã sửa v1):
    Canvas nền đen(26) + nét trắng(255) đã đúng chuẩn MNIST.
    Bản gốc đảo 255-img → nét trắng thành 0, bị threshold xóa mất.

  LỖI 2 — Threshold sai chiều (đã sửa v1):
    Bản gốc: img < 50 → 0  (xóa luôn nét vẽ đã bị đảo).
    Bản sửa: img > 50 → 255 (giữ nét vẽ đúng).

  LỖI 3 — DOMAIN GAP với MNIST (sửa v3 — lý do vẫn sai sau v1):
    Logistic Regression được train trên ảnh MNIST với quy trình:
    (a) Nội dung chữ số nằm trong vùng 20×20, border 4px
    (b) Căn giữa theo CENTER OF MASS (không phải bbox center)
    (c) Pixel là grayscale mịn (anti-aliased), không phải binary
    Sau v1, input vẫn dùng bbox centering + pixel gần binary
    → phân phối khác với lúc train → model nhận sai.

  ✅ V3 ÁP DỤNG ĐÚNG QUY TRÌNH CHUẨN MNIST:
    1. Resize nội dung về 20×20 → đặt vào 28×28 (border 4px)
    2. Dịch chuyển theo center of mass → căn giữa chính xác
    3. Gaussian blur nhẹ (σ=0.5) → mô phỏng anti-aliasing MNIST
    4. Normalize float [0,1] — không binary
    Yêu cầu thêm: pip install scipy
================================================
"""

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw
import numpy as np
import joblib

try:
    from scipy.ndimage import gaussian_filter, center_of_mass, shift as nd_shift
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy chưa cài. Chạy: pip install scipy")

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_PATH         = "digit_model.pkl"
CANVAS_SIZE        = 300
BRUSH_RADIUS       = 14
BG_COLOR           = "#1a1a1a"
FG_COLOR           = "white"
AUTO_PREDICT_DELAY = 800

# ============================================================
# TẢI MÔ HÌNH
# ============================================================
print("Đang tải mô hình...")
model = joblib.load(MODEL_PATH)
print("Tải mô hình thành công!")

# ============================================================
# TIỀN XỬ LÝ ẢNH — CHUẨN MNIST v3
# ============================================================
def preprocess(pil_image):
    """
    Chuyển ảnh vẽ tay → vector 784 chiều đúng chuẩn MNIST.

    Quy trình (theo chuẩn sklearn/LeCun MNIST):
      1. Grayscale
      2. Threshold pixel > 50 → giữ nét vẽ
      3. Crop bounding box + padding 15%
      4. Pad về hình vuông
      5. Resize về 20×20
      6. Đặt vào canvas 28×28 (border 4px như MNIST)
      7. Dịch theo center of mass về tâm (14,14) [QUAN TRỌNG]
      8. Gaussian blur nhẹ σ=0.5 → mô phỏng anti-aliasing MNIST
      9. Normalize về [0,1]
    """
    img = np.array(pil_image.convert('L'), dtype=np.float32)

    # Threshold: giữ nét vẽ sáng
    binary = (img > 50).astype(np.uint8)

    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad = max(5, int(max(y_max - y_min, x_max - x_min) * 0.15))
    y_min = max(0, y_min - pad)
    y_max = min(img.shape[0] - 1, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(img.shape[1] - 1, x_max + pad)
    crop = img[y_min:y_max + 1, x_min:x_max + 1]

    # Pad về hình vuông
    h, w = crop.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.float32)
    square[(size - h) // 2:(size - h) // 2 + h,
           (size - w) // 2:(size - w) // 2 + w] = crop

    # Resize 20×20 → đặt vào 28×28 với border 4px
    pil_20 = Image.fromarray(square.astype(np.uint8)).resize(
        (20, 20), Image.LANCZOS)
    arr20 = np.array(pil_20, dtype=np.float32)

    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[4:24, 4:24] = arr20

    # Căn giữa theo center of mass (chuẩn MNIST)
    if HAS_SCIPY and canvas28.sum() > 0:
        cy, cx = center_of_mass(canvas28)
        dy = int(round(14 - cy))
        dx = int(round(14 - cx))
        canvas28 = nd_shift(canvas28, [dy, dx], mode='constant', cval=0)
        # Blur nhẹ mô phỏng anti-aliasing của MNIST
        canvas28 = gaussian_filter(canvas28, sigma=0.5)

    if canvas28.max() > 0:
        canvas28 = canvas28 / canvas28.max()

    return canvas28.reshape(1, -1)


# ============================================================
# GIAO DIỆN TKINTER
# ============================================================
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận dạng chữ số viết tay — Đề tài 18")
        self.root.configure(bg="#f5f5f5")
        self.root.resizable(False, False)

        self.auto_predict_var  = tk.BooleanVar(value=True)
        self._auto_predict_job = None

        title_font  = tkfont.Font(family="Helvetica", size=14, weight="bold")
        label_font  = tkfont.Font(family="Helvetica", size=11)
        result_font = tkfont.Font(family="Courier",   size=48, weight="bold")
        small_font  = tkfont.Font(family="Helvetica", size=9)
        hint_font   = tkfont.Font(family="Helvetica", size=8,  slant="italic")

        # ── Tiêu đề ──────────────────────────────────────────
        header = tk.Frame(root, bg="#1a1a1a", pady=10)
        header.pack(fill=tk.X)
        tk.Label(header, text="✍  Nhận dạng chữ số viết tay",
                 font=title_font, bg="#1a1a1a", fg="white").pack()
        tk.Label(header, text="Vẽ một chữ số (0–9) vào ô bên dưới",
                 font=small_font, bg="#1a1a1a", fg="#aaa").pack()

        # ── Khu vực chính ────────────────────────────────────
        content = tk.Frame(root, bg="#f5f5f5", padx=16, pady=16)
        content.pack()

        left = tk.Frame(content, bg="#f5f5f5")
        left.grid(row=0, column=0, padx=(0, 16))

        tk.Label(left, text="Bảng vẽ", font=label_font,
                 bg="#f5f5f5", fg="#333").pack(anchor="w", pady=(0, 4))
        tk.Label(left,
                 text="💡 Vẽ to, rõ, ở giữa ô — nét càng tự nhiên càng chính xác",
                 font=hint_font, bg="#f5f5f5", fg="#888").pack(
                     anchor="w", pady=(0, 4))

        border = tk.Frame(left, bg="#333", padx=2, pady=2)
        border.pack()

        self.canvas = tk.Canvas(border,
                                width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg=BG_COLOR, cursor="pencil",
                                highlightthickness=0)
        self.canvas.pack()

        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        self.draw  = ImageDraw.Draw(self.image)

        self.last_x = self.last_y = None
        self.canvas.bind("<Button-1>",        self.on_click)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Nút bấm
        btn_frame = tk.Frame(left, bg="#f5f5f5")
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Button(btn_frame, text="Xóa", font=label_font,
                  bg="white", fg="#333", relief="flat",
                  bd=1, padx=16, pady=6,
                  command=self.clear).pack(
                      side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(btn_frame, text="Nhận dạng", font=label_font,
                  bg="#1a1a1a", fg="white", relief="flat",
                  padx=16, pady=6,
                  command=self.predict).pack(
                      side=tk.RIGHT, expand=True,
                      fill=tk.X, padx=(8, 0))

        auto_frame = tk.Frame(left, bg="#f5f5f5")
        auto_frame.pack(fill=tk.X, pady=(6, 0))
        tk.Checkbutton(
            auto_frame,
            text=f"Tự động nhận dạng sau {AUTO_PREDICT_DELAY}ms khi ngừng vẽ",
            variable=self.auto_predict_var,
            font=hint_font, bg="#f5f5f5", fg="#666",
            activebackground="#f5f5f5").pack(anchor="w")

        # ── Khu vực kết quả ──────────────────────────────────
        right = tk.Frame(content, bg="white", bd=0,
                         highlightthickness=1, highlightbackground="#ddd",
                         width=200, pady=12, padx=16)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)

        tk.Label(right, text="Kết quả", font=label_font,
                 bg="white", fg="#666").pack(anchor="w")

        self.result_digit = tk.Label(right, text="?",
                                     font=result_font,
                                     bg="white", fg="#1a1a1a")
        self.result_digit.pack(pady=(8, 0))

        self.result_conf = tk.Label(right, text="Vẽ để bắt đầu",
                                    font=small_font, bg="white", fg="#888")
        self.result_conf.pack()

        tk.Label(right, text="Xác suất từng chữ số",
                 font=small_font, bg="white", fg="#999").pack(
                     anchor="w", pady=(16, 4))

        self.bars       = {}
        self.bar_labels = {}

        for d in range(10):
            row_f = tk.Frame(right, bg="white")
            row_f.pack(fill=tk.X, pady=1)
            tk.Label(row_f, text=str(d), font=small_font,
                     bg="white", fg="#555", width=2, anchor="e").pack(side=tk.LEFT)
            track = tk.Frame(row_f, bg="#eee", height=10)
            track.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            track.pack_propagate(False)
            bar = tk.Frame(track, bg="#bbb", height=10)
            bar.place(x=0, y=0, relwidth=0, height=10)
            lbl = tk.Label(row_f, text="0%", font=small_font,
                           bg="white", fg="#999", width=4, anchor="w")
            lbl.pack(side=tk.LEFT)
            self.bars[d]       = (track, bar)
            self.bar_labels[d] = lbl

        # Status bar
        scipy_ok = HAS_SCIPY
        self.status = tk.Label(
            root,
            text="Sẵn sàng | " + ("✅ scipy" if scipy_ok else "⚠️  pip install scipy"),
            font=small_font, bg="#eee",
            fg="#2e7d32" if scipy_ok else "#e65100",
            anchor="w", padx=10, pady=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Sự kiện vẽ ───────────────────────────────────────────
    def on_click(self, e):
        self.last_x, self.last_y = e.x, e.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r,
                                fill=FG_COLOR, outline=FG_COLOR)
        self.draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=FG_COLOR)
        self._cancel_auto_predict()

    def on_drag(self, e):
        if self.last_x is None:
            return
        r  = BRUSH_RADIUS
        dx = e.x - self.last_x
        dy = e.y - self.last_y
        steps = max(1, int((dx**2 + dy**2)**0.5 / (r * 0.5)))
        for i in range(steps + 1):
            t  = i / steps
            ix = int(self.last_x + dx * t)
            iy = int(self.last_y + dy * t)
            self.canvas.create_oval(ix-r, iy-r, ix+r, iy+r,
                                    fill=FG_COLOR, outline=FG_COLOR)
            self.draw.ellipse([ix-r, iy-r, ix+r, iy+r], fill=FG_COLOR)
        self.last_x, self.last_y = e.x, e.y
        self._cancel_auto_predict()

    def on_release(self, e):
        self.last_x = self.last_y = None
        if self.auto_predict_var.get():
            self._schedule_auto_predict()

    def _schedule_auto_predict(self):
        self._cancel_auto_predict()
        self._auto_predict_job = self.root.after(AUTO_PREDICT_DELAY, self.predict)

    def _cancel_auto_predict(self):
        if self._auto_predict_job is not None:
            self.root.after_cancel(self._auto_predict_job)
            self._auto_predict_job = None

    def clear(self):
        self._cancel_auto_predict()
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=BG_COLOR)
        self.result_digit.config(text="?", fg="#1a1a1a")
        self.result_conf.config(text="Vẽ để bắt đầu", fg="#888")
        self._reset_bars()
        self.status.config(text="Đã xóa — sẵn sàng vẽ lại")

    def _reset_bars(self):
        for d in range(10):
            _, bar = self.bars[d]
            bar.place_configure(relwidth=0)
            bar.config(bg="#bbb")
            self.bar_labels[d].config(text="0%", fg="#999")

    def predict(self):
        self._cancel_auto_predict()
        self.status.config(text="Đang nhận dạng...")
        self.root.update()

        X = preprocess(self.image)
        if X is None:
            self.status.config(text="⚠️  Canvas trống — hãy vẽ một chữ số!")
            return

        pred   = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        conf   = float(np.max(probas))

        conf_color = "#2e7d32" if conf >= 0.80 else ("#e65100" if conf >= 0.60 else "#c62828")

        self.result_digit.config(text=str(pred), fg="#1a1a1a")
        self.result_conf.config(text=f"Độ tin cậy: {conf:.1%}", fg=conf_color)

        self._reset_bars()
        for d in range(10):
            p = float(probas[d])
            _, bar = self.bars[d]
            bar.config(bg="#1a1a1a" if d == pred else "#90caf9")
            bar.place_configure(relwidth=p)
            self.bar_labels[d].config(
                text=f"{p:.0%}",
                fg="#1a1a1a" if d == pred else "#666")

        self.status.config(text=f"Kết quả: chữ số {pred} | Tin cậy: {conf:.2%}")


# ============================================================
# CHẠY ỨNG DỤNG
# ============================================================
if __name__ == "__main__":
    if not HAS_SCIPY:
        print("\n" + "="*50)
        print("  KHUYẾN NGHỊ: pip install scipy")
        print("  (Cần thiết để căn giữa center-of-mass)")
        print("="*50 + "\n")
    root = tk.Tk()
    app  = DigitApp(root)
    root.mainloop()