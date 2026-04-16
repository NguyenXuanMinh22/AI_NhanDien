"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  draw_predict.py — v4 (+ Accuracy Display)
================================================
"""

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw
import numpy as np
import joblib
import threading

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
    h, w  = crop.shape; size = max(h, w)
    square = np.zeros((size, size), dtype=np.float32)
    square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = crop
    pil_20   = Image.fromarray(square.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[4:24, 4:24] = np.array(pil_20, dtype=np.float32)
    if HAS_SCIPY and canvas28.sum() > 0:
        cy, cx   = center_of_mass(canvas28)
        canvas28 = nd_shift(canvas28, [int(round(14-cy)), int(round(14-cx))],
                            mode='constant', cval=0)
        canvas28 = gaussian_filter(canvas28, sigma=0.5)
    if canvas28.max() > 0:
        canvas28 = canvas28 / canvas28.max()
    return canvas28.reshape(1, -1)


# ============================================================
# CỬA SỔ ACCURACY
# ============================================================
class AccuracyWindow:
    """
    Cửa sổ riêng hiển thị accuracy của model trên MNIST.
    Tính toán chạy trên thread riêng để không đóng băng UI.
    Hiển thị:
      • Độ chính xác tổng thể (số lớn, màu theo ngưỡng)
      • Thanh accuracy từng chữ số 0–9 (màu xanh/cam/đỏ)
      • Ma trận nhầm lẫn 10×10 dạng heatmap (tkinter Canvas)
    """

    # Màu cho confusion matrix (gradient trắng → xanh đậm)
    CM_COLORS = [
        "#ffffff", "#e3f2fd", "#bbdefb", "#90caf9",
        "#64b5f6", "#42a5f5", "#2196f3", "#1976d2",
        "#1565c0", "#0d47a1",
    ]

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("📊 Accuracy — Đánh giá mô hình")
        self.win.configure(bg="#f5f5f5")
        self.win.resizable(False, False)
        self.win.grab_set()           # modal
        self.win.focus_set()

        # Fonts
        self.f_title  = tkfont.Font(family="Helvetica", size=13, weight="bold")
        self.f_big    = tkfont.Font(family="Courier",   size=42, weight="bold")
        self.f_label  = tkfont.Font(family="Helvetica", size=10)
        self.f_small  = tkfont.Font(family="Helvetica", size=8)
        self.f_hint   = tkfont.Font(family="Helvetica", size=8, slant="italic")
        self.f_cm     = tkfont.Font(family="Courier",   size=7)

        self._build_ui()
        # Bắt đầu tính toán trên thread nền
        threading.Thread(target=self._compute, daemon=True).start()

    # ── Xây dựng UI ──────────────────────────────────────────
    def _build_ui(self):
        w = self.win

        # Header
        hdr = tk.Frame(w, bg="#1a1a1a", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="📊  Đánh giá mô hình trên MNIST",
                 font=self.f_title, bg="#1a1a1a", fg="white").pack()
        tk.Label(hdr, text="Đang tải dữ liệu MNIST và tính toán...",
                 font=self.f_hint, bg="#1a1a1a", fg="#aaa").pack()
        self._hdr_sub = hdr.winfo_children()[-1]   # lưu để cập nhật sau

        main = tk.Frame(w, bg="#f5f5f5", padx=16, pady=14)
        main.pack()

        # ── Cột trái: tổng + từng số ─────────────────────────
        left = tk.Frame(main, bg="#f5f5f5")
        left.grid(row=0, column=0, sticky="n", padx=(0, 18))

        # Accuracy tổng thể
        box = tk.Frame(left, bg="white", highlightthickness=1,
                       highlightbackground="#ddd", padx=20, pady=12)
        box.pack(fill=tk.X, pady=(0, 12))
        tk.Label(box, text="Độ chính xác tổng thể",
                 font=self.f_label, bg="white", fg="#666").pack()
        self._lbl_overall = tk.Label(box, text="—",
                                     font=self.f_big,
                                     bg="white", fg="#bbb")
        self._lbl_overall.pack()
        self._lbl_samples = tk.Label(box, text="Đang tính...",
                                     font=self.f_hint, bg="white", fg="#aaa")
        self._lbl_samples.pack()

        # Accuracy từng chữ số
        per = tk.Frame(left, bg="white", highlightthickness=1,
                       highlightbackground="#ddd", padx=14, pady=10)
        per.pack(fill=tk.X)
        tk.Label(per, text="Accuracy từng chữ số",
                 font=self.f_label, bg="white", fg="#666").pack(anchor="w", pady=(0,6))

        self._digit_bars   = {}
        self._digit_labels = {}
        BAR_W = 180

        for d in range(10):
            row = tk.Frame(per, bg="white")
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=f"  {d} ", font=self.f_label,
                     bg="white", fg="#333", width=3,
                     anchor="e").pack(side=tk.LEFT)

            track = tk.Frame(row, bg="#eee", height=14, width=BAR_W)
            track.pack(side=tk.LEFT, padx=(4,0))
            track.pack_propagate(False)

            bar = tk.Frame(track, bg="#bbb", height=14)
            bar.place(x=0, y=0, width=0, height=14)

            pct_lbl = tk.Label(row, text="—", font=self.f_small,
                               bg="white", fg="#aaa", width=6, anchor="w")
            pct_lbl.pack(side=tk.LEFT, padx=(5,0))

            self._digit_bars[d]   = (track, bar, BAR_W)
            self._digit_labels[d] = pct_lbl

        # ── Cột phải: confusion matrix ────────────────────────
        right = tk.Frame(main, bg="#f5f5f5")
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="Ma trận nhầm lẫn (Confusion Matrix)",
                 font=self.f_label, bg="#f5f5f5", fg="#666").pack(anchor="w", pady=(0,6))
        tk.Label(right,
                 text="Hàng = Thực tế  |  Cột = Dự đoán  |  Đường chéo = Đúng",
                 font=self.f_hint, bg="#f5f5f5", fg="#999").pack(anchor="w", pady=(0,4))

        CM_CELL = 36   # pixel mỗi ô
        CM_N    = 10
        cm_total_w = CM_CELL * (CM_N + 1) + 2
        cm_total_h = CM_CELL * (CM_N + 1) + 2

        self._cm_canvas = tk.Canvas(right,
                                    width=cm_total_w, height=cm_total_h,
                                    bg="#f5f5f5", highlightthickness=0)
        self._cm_canvas.pack()
        self._CM_CELL = CM_CELL
        self._draw_cm_placeholder()

        # Nút đóng
        tk.Button(w, text="Đóng", font=self.f_label,
                  bg="#1a1a1a", fg="white", relief="flat",
                  padx=24, pady=6,
                  command=self.win.destroy).pack(pady=(0, 14))

    # ── Vẽ placeholder cho confusion matrix ──────────────────
    def _draw_cm_placeholder(self):
        c   = self._cm_canvas
        sz  = self._CM_CELL
        c.delete("all")
        # Header hàng/cột
        for i in range(10):
            cx = sz + i * sz + sz // 2
            cy = sz // 2
            c.create_text(cx, cy, text=str(i),
                          font=self.f_cm, fill="#999")
            c.create_text(sz // 2, sz + i * sz + sz // 2,
                          text=str(i), font=self.f_cm, fill="#999")
        # Ô placeholder
        for r in range(10):
            for col in range(10):
                x0 = sz + col * sz + 1
                y0 = sz + r   * sz + 1
                c.create_rectangle(x0, y0, x0+sz-2, y0+sz-2,
                                   fill="#eee", outline="#ddd")

    # ── Tính toán trên thread nền ─────────────────────────────
    def _compute(self):
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.metrics import accuracy_score, confusion_matrix

            # Cập nhật status
            self.win.after(0, lambda: self._hdr_sub.config(
                text="Đang tải MNIST (lần đầu có thể mất ~30s)..."))

            mnist = fetch_openml('mnist_784', version=1,
                                 as_frame=False, cache=True)
            X = mnist.data / 255.0
            y = mnist.target.astype(int)

            # Dùng 10,000 mẫu cuối làm test set (không overlap với train thường)
            X_eval = X[-10000:]
            y_eval = y[-10000:]

            self.win.after(0, lambda: self._hdr_sub.config(
                text="Đang dự đoán 10,000 mẫu..."))

            y_pred = model.predict(X_eval)
            acc    = accuracy_score(y_eval, y_pred)
            cm     = confusion_matrix(y_eval, y_pred)

            # Accuracy từng chữ số
            per_acc = cm.diagonal() / cm.sum(axis=1)

            # Cập nhật UI trên main thread
            self.win.after(0, lambda: self._update_ui(
                acc, per_acc, cm, len(y_eval)))

        except Exception as e:
            self.win.after(0, lambda: self._show_error(str(e)))

    # ── Cập nhật UI sau khi tính xong ────────────────────────
    def _update_ui(self, acc, per_acc, cm, n_samples):
        # Header
        self._hdr_sub.config(
            text=f"Đánh giá trên {n_samples:,} mẫu MNIST (test set)")

        # Accuracy tổng thể
        pct = acc * 100
        color = ("#2e7d32" if pct >= 95
                 else "#e65100" if pct >= 90
                 else "#c62828")
        self._lbl_overall.config(text=f"{pct:.2f}%", fg=color)
        self._lbl_samples.config(
            text=f"{int(acc * n_samples):,} / {n_samples:,} mẫu đúng",
            fg="#666")

        # Accuracy từng chữ số
        for d in range(10):
            track, bar, bar_w = self._digit_bars[d]
            a = per_acc[d]
            w = int(a * bar_w)

            bar_color = ("#2e7d32" if a >= 0.95
                         else "#e65100" if a >= 0.90
                         else "#c62828")
            bar.config(bg=bar_color)
            bar.place_configure(width=w)

            lbl_color = bar_color
            self._digit_labels[d].config(
                text=f"{a*100:.1f}%", fg=lbl_color)

        # Confusion matrix
        self._draw_cm(cm)

    # ── Vẽ confusion matrix ───────────────────────────────────
    def _draw_cm(self, cm):
        c   = self._cm_canvas
        sz  = self._CM_CELL
        c.delete("all")

        row_totals = cm.sum(axis=1, keepdims=True).clip(1)
        cm_norm    = cm / row_totals   # normalize theo hàng

        # Header nhãn
        for i in range(10):
            # Cột header (dự đoán)
            c.create_text(sz + i*sz + sz//2, sz//2,
                          text=str(i), font=self.f_cm, fill="#555")
            # Hàng header (thực tế)
            c.create_text(sz//2, sz + i*sz + sz//2,
                          text=str(i), font=self.f_cm, fill="#555")

        # Ô dữ liệu
        for r in range(10):
            for col in range(10):
                val  = cm[r, col]
                norm = cm_norm[r, col]

                # Màu nền: gradient theo giá trị normalize
                ci       = min(9, int(norm * 9.99))
                bg_color = self.CM_COLORS[ci]

                x0 = sz + col*sz + 1
                y0 = sz + r  *sz + 1
                x1 = x0 + sz - 2
                y1 = y0 + sz - 2

                # Highlight đường chéo (dự đoán đúng)
                outline = "#1976d2" if r == col else "#e0e0e0"
                c.create_rectangle(x0, y0, x1, y1,
                                   fill=bg_color, outline=outline,
                                   width=2 if r == col else 1)

                # Chữ số trong ô
                txt_color = "white" if norm > 0.55 else "#1a1a1a"
                c.create_text((x0+x1)//2, (y0+y1)//2,
                              text=str(val),
                              font=self.f_cm, fill=txt_color)

        # Chú thích màu
        legend_y = sz + 10*sz + 6
        c.create_text(sz, legend_y, anchor="w",
                      text="■ Đúng (đường chéo)  □ Sai (ngoài chéo)",
                      font=self.f_hint, fill="#888")

    # ── Hiển thị lỗi ─────────────────────────────────────────
    def _show_error(self, msg):
        self._hdr_sub.config(text=f"❌ Lỗi: {msg}", fg="#c62828")
        self._lbl_overall.config(text="Lỗi", fg="#c62828")
        self._lbl_samples.config(
            text="Không thể tải MNIST.\nKiểm tra kết nối internet.", fg="#c62828")


# ============================================================
# GIAO DIỆN CHÍNH
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

        # ── Hàng nút chính ───────────────────────────────────
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
                      side=tk.LEFT, expand=True,
                      fill=tk.X, padx=(8, 0))

        # ── Nút Accuracy ─────────────────────────────────────
        tk.Button(btn_frame, text="📊 Accuracy", font=label_font,
                  bg="#1565c0", fg="white", relief="flat",
                  padx=10, pady=6,
                  command=self.show_accuracy).pack(
                      side=tk.RIGHT, expand=True,
                      fill=tk.X, padx=(8, 0))

        # Auto-predict checkbox
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

        # ── Status bar ───────────────────────────────────────
        scipy_ok = HAS_SCIPY
        self.status = tk.Label(
            root,
            text="Sẵn sàng | " + ("✅ scipy" if scipy_ok else "⚠️  pip install scipy"),
            font=small_font, bg="#eee",
            fg="#2e7d32" if scipy_ok else "#e65100",
            anchor="w", padx=10, pady=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Hiển thị cửa sổ Accuracy ─────────────────────────────
    def show_accuracy(self):
        AccuracyWindow(self.root)

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

        conf_color = ("#2e7d32" if conf >= 0.80
                      else "#e65100" if conf >= 0.60
                      else "#c62828")

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

        self.status.config(
            text=f"Kết quả: chữ số {pred} | Tin cậy: {conf:.2%}")


# ============================================================
# CHẠY ỨNG DỤNG
# ============================================================
if __name__ == "__main__":
    if not HAS_SCIPY:
        print("\n" + "="*50)
        print("  KHUYẾN NGHỊ: pip install scipy")
        print("="*50 + "\n")
    root = tk.Tk()
    app  = DigitApp(root)
    root.mainloop()