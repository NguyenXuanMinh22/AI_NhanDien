"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  draw_predict.py — v5 (+ Upload ảnh)
================================================

  TÍNH NĂNG:
    • Tab "✍ Vẽ tay"  — vẽ chữ số trực tiếp lên canvas
    • Tab "📁 Upload"  — tải ảnh từ máy tính để dự đoán
    • Nút "📊 Accuracy" — đánh giá model trên MNIST
    • Auto-predict sau 800ms khi ngừng vẽ

  UPLOAD ẢNH:
    • Hỗ trợ: PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP
    • Tự nhận diện nền sáng/tối và đảo màu nếu cần
    • Hiển thị ảnh gốc + preview 28×28 sau xử lý
    • Hỗ trợ kéo-thả file (drag & drop)
================================================
"""

import tkinter as tk
from tkinter import font as tkfont, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import joblib
import threading
import os

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

UPLOAD_FILETYPES = [
    ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp"),
    ("PNG",  "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("Tất cả", "*.*"),
]

# ============================================================
# TẢI MÔ HÌNH
# ============================================================
print("Đang tải mô hình...")
model = joblib.load(MODEL_PATH)
print("Tải mô hình thành công!")

# ============================================================
# TIỀN XỬ LÝ ẢNH — CHUẨN MNIST (dùng chung cho vẽ & upload)
# ============================================================
def _to_mnist_tensor(gray_float):
    """
    gray_float: np.array float32, shape (H,W), pixel sáng = chữ số.
    Trả về vector (1, 784) chuẩn MNIST hoặc None nếu canvas trống.
    """
    binary = (gray_float > 50).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad   = max(5, int(max(y_max - y_min, x_max - x_min) * 0.15))
    y_min = max(0, y_min - pad);   y_max = min(gray_float.shape[0]-1, y_max + pad)
    x_min = max(0, x_min - pad);   x_max = min(gray_float.shape[1]-1, x_max + pad)
    crop  = gray_float[y_min:y_max+1, x_min:x_max+1]

    h, w = crop.shape; size = max(h, w)
    square = np.zeros((size, size), dtype=np.float32)
    square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = crop

    pil_20   = Image.fromarray(square.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[4:24, 4:24] = np.array(pil_20, dtype=np.float32)

    if HAS_SCIPY and canvas28.sum() > 0:
        cy, cx   = center_of_mass(canvas28)
        canvas28 = nd_shift(canvas28,
                            [int(round(14-cy)), int(round(14-cx))],
                            mode='constant', cval=0)
        canvas28 = gaussian_filter(canvas28, sigma=0.5)

    if canvas28.max() > 0:
        canvas28 = canvas28 / canvas28.max()

    return canvas28.reshape(1, -1), canvas28   # (vector, 28x28 preview)


def preprocess(pil_image):
    """Dành cho canvas vẽ tay: nền tối, nét sáng."""
    img = np.array(pil_image.convert('L'), dtype=np.float32)
    result = _to_mnist_tensor(img)
    if result is None:
        return None
    return result[0]   # chỉ trả vector


def preprocess_upload(pil_image):
    """
    Dành cho ảnh upload: tự động phát hiện nền sáng/tối.

    Quy tắc: nếu median pixel > 127 → ảnh nền sáng, chữ tối
             → đảo màu để chữ thành sáng (chuẩn MNIST).
    Trả về (vector_784, arr_28x28, was_inverted).
    """
    img = np.array(pil_image.convert('L'), dtype=np.float32)

    # Tự nhận diện nền sáng / tối
    median_val = float(np.median(img))
    inverted   = False
    if median_val > 127:          # nền sáng → đảo để chữ thành sáng
        img      = 255.0 - img
        inverted = True

    result = _to_mnist_tensor(img)
    if result is None:
        return None, None, inverted

    vec, arr28 = result
    return vec, arr28, inverted


def arr28_to_photoimage(arr28, size=84):
    """Chuyển mảng 28×28 float[0,1] → PhotoImage để hiển thị tkinter."""
    pixels = (arr28 * 255).astype(np.uint8)
    pil    = Image.fromarray(pixels, mode='L').resize(
        (size, size), Image.NEAREST)
    return ImageTk.PhotoImage(pil)


# ============================================================
# CỬA SỔ ACCURACY
# ============================================================
class AccuracyWindow:
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
        self.win.grab_set()
        self.win.focus_set()

        self.f_title = tkfont.Font(family="Helvetica", size=13, weight="bold")
        self.f_big   = tkfont.Font(family="Courier",   size=42, weight="bold")
        self.f_label = tkfont.Font(family="Helvetica", size=10)
        self.f_small = tkfont.Font(family="Helvetica", size=8)
        self.f_hint  = tkfont.Font(family="Helvetica", size=8, slant="italic")
        self.f_cm    = tkfont.Font(family="Courier",   size=7)

        self._build_ui()
        threading.Thread(target=self._compute, daemon=True).start()

    def _build_ui(self):
        w = self.win
        hdr = tk.Frame(w, bg="#1a1a1a", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="📊  Đánh giá mô hình trên MNIST",
                 font=self.f_title, bg="#1a1a1a", fg="white").pack()
        tk.Label(hdr, text="Đang tải dữ liệu MNIST và tính toán...",
                 font=self.f_hint, bg="#1a1a1a", fg="#aaa").pack()
        self._hdr_sub = hdr.winfo_children()[-1]

        main = tk.Frame(w, bg="#f5f5f5", padx=16, pady=14)
        main.pack()

        left = tk.Frame(main, bg="#f5f5f5")
        left.grid(row=0, column=0, sticky="n", padx=(0, 18))

        box = tk.Frame(left, bg="white", highlightthickness=1,
                       highlightbackground="#ddd", padx=20, pady=12)
        box.pack(fill=tk.X, pady=(0, 12))
        tk.Label(box, text="Độ chính xác tổng thể",
                 font=self.f_label, bg="white", fg="#666").pack()
        self._lbl_overall = tk.Label(box, text="—", font=self.f_big,
                                     bg="white", fg="#bbb")
        self._lbl_overall.pack()
        self._lbl_samples = tk.Label(box, text="Đang tính...",
                                     font=self.f_hint, bg="white", fg="#aaa")
        self._lbl_samples.pack()

        per = tk.Frame(left, bg="white", highlightthickness=1,
                       highlightbackground="#ddd", padx=14, pady=10)
        per.pack(fill=tk.X)
        tk.Label(per, text="Accuracy từng chữ số",
                 font=self.f_label, bg="white", fg="#666").pack(
                     anchor="w", pady=(0, 6))

        self._digit_bars   = {}
        self._digit_labels = {}
        BAR_W = 180

        for d in range(10):
            row = tk.Frame(per, bg="white")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"  {d} ", font=self.f_label,
                     bg="white", fg="#333", width=3, anchor="e").pack(side=tk.LEFT)
            track = tk.Frame(row, bg="#eee", height=14, width=BAR_W)
            track.pack(side=tk.LEFT, padx=(4, 0))
            track.pack_propagate(False)
            bar = tk.Frame(track, bg="#bbb", height=14)
            bar.place(x=0, y=0, width=0, height=14)
            lbl = tk.Label(row, text="—", font=self.f_small,
                           bg="white", fg="#aaa", width=6, anchor="w")
            lbl.pack(side=tk.LEFT, padx=(5, 0))
            self._digit_bars[d]   = (track, bar, BAR_W)
            self._digit_labels[d] = lbl

        right = tk.Frame(main, bg="#f5f5f5")
        right.grid(row=0, column=1, sticky="n")
        tk.Label(right, text="Ma trận nhầm lẫn (Confusion Matrix)",
                 font=self.f_label, bg="#f5f5f5", fg="#666").pack(
                     anchor="w", pady=(0, 6))
        tk.Label(right,
                 text="Hàng = Thực tế  |  Cột = Dự đoán  |  Đường chéo = Đúng",
                 font=self.f_hint, bg="#f5f5f5", fg="#999").pack(
                     anchor="w", pady=(0, 4))

        CM_CELL = 36
        cm_w    = CM_CELL * 11 + 2
        self._cm_canvas = tk.Canvas(right, width=cm_w, height=cm_w,
                                    bg="#f5f5f5", highlightthickness=0)
        self._cm_canvas.pack()
        self._CM_CELL = CM_CELL
        self._draw_cm_placeholder()

        tk.Button(w, text="Đóng", font=self.f_label,
                  bg="#1a1a1a", fg="white", relief="flat",
                  padx=24, pady=6,
                  command=self.win.destroy).pack(pady=(0, 14))

    def _draw_cm_placeholder(self):
        c = self._cm_canvas; sz = self._CM_CELL
        c.delete("all")
        for i in range(10):
            c.create_text(sz + i*sz + sz//2, sz//2,
                          text=str(i), font=self.f_cm, fill="#999")
            c.create_text(sz//2, sz + i*sz + sz//2,
                          text=str(i), font=self.f_cm, fill="#999")
        for r in range(10):
            for col in range(10):
                x0 = sz + col*sz + 1; y0 = sz + r*sz + 1
                c.create_rectangle(x0, y0, x0+sz-2, y0+sz-2,
                                   fill="#eee", outline="#ddd")

    def _compute(self):
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.metrics import accuracy_score, confusion_matrix

            self.win.after(0, lambda: self._hdr_sub.config(
                text="Đang tải MNIST (lần đầu có thể mất ~30s)..."))
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
            X = mnist.data / 255.0
            y = mnist.target.astype(int)
            X_eval = X[-10000:]; y_eval = y[-10000:]

            self.win.after(0, lambda: self._hdr_sub.config(
                text="Đang dự đoán 10,000 mẫu..."))
            y_pred = model.predict(X_eval)
            acc    = accuracy_score(y_eval, y_pred)
            cm     = confusion_matrix(y_eval, y_pred)
            per_acc = cm.diagonal() / cm.sum(axis=1)

            self.win.after(0, lambda: self._update_ui(acc, per_acc, cm, len(y_eval)))
        except Exception as e:
            self.win.after(0, lambda: self._show_error(str(e)))

    def _update_ui(self, acc, per_acc, cm, n_samples):
        self._hdr_sub.config(text=f"Đánh giá trên {n_samples:,} mẫu MNIST (test set)")
        pct   = acc * 100
        color = "#2e7d32" if pct >= 95 else "#e65100" if pct >= 90 else "#c62828"
        self._lbl_overall.config(text=f"{pct:.2f}%", fg=color)
        self._lbl_samples.config(
            text=f"{int(acc*n_samples):,} / {n_samples:,} mẫu đúng", fg="#666")
        for d in range(10):
            track, bar, bw = self._digit_bars[d]
            a = per_acc[d]
            bc = "#2e7d32" if a >= 0.95 else "#e65100" if a >= 0.90 else "#c62828"
            bar.config(bg=bc); bar.place_configure(width=int(a * bw))
            self._digit_labels[d].config(text=f"{a*100:.1f}%", fg=bc)
        self._draw_cm(cm)

    def _draw_cm(self, cm):
        c = self._cm_canvas; sz = self._CM_CELL
        c.delete("all")
        row_totals = cm.sum(axis=1, keepdims=True).clip(1)
        cm_norm    = cm / row_totals
        for i in range(10):
            c.create_text(sz + i*sz + sz//2, sz//2,
                          text=str(i), font=self.f_cm, fill="#555")
            c.create_text(sz//2, sz + i*sz + sz//2,
                          text=str(i), font=self.f_cm, fill="#555")
        for r in range(10):
            for col in range(10):
                val  = cm[r, col]
                norm = cm_norm[r, col]
                ci   = min(9, int(norm * 9.99))
                bg   = self.CM_COLORS[ci]
                x0   = sz + col*sz + 1; y0 = sz + r*sz + 1
                x1   = x0 + sz - 2;     y1 = y0 + sz - 2
                outline = "#1976d2" if r == col else "#e0e0e0"
                c.create_rectangle(x0, y0, x1, y1, fill=bg, outline=outline,
                                   width=2 if r == col else 1)
                tc = "white" if norm > 0.55 else "#1a1a1a"
                c.create_text((x0+x1)//2, (y0+y1)//2,
                              text=str(val), font=self.f_cm, fill=tc)

    def _show_error(self, msg):
        self._hdr_sub.config(text=f"❌ Lỗi: {msg}", fg="#c62828")
        self._lbl_overall.config(text="Lỗi", fg="#c62828")
        self._lbl_samples.config(
            text="Không tải được MNIST.\nKiểm tra kết nối internet.", fg="#c62828")


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
        self._current_tab      = "draw"    # "draw" | "upload"
        self._upload_photo     = None      # giữ reference PhotoImage
        self._preview_photo    = None      # 28×28 preview PhotoImage

        # Fonts
        self.f_title  = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self.f_label  = tkfont.Font(family="Helvetica", size=11)
        self.f_result = tkfont.Font(family="Courier",   size=48, weight="bold")
        self.f_small  = tkfont.Font(family="Helvetica", size=9)
        self.f_hint   = tkfont.Font(family="Helvetica", size=8,  slant="italic")

        self._build_header()
        self._build_statusbar()   # ← phải tạo trước để _switch_tab dùng được
        self._build_content()

        # Drag & Drop trên Windows/Linux
        try:
            self.root.drop_target_register('DND_Files')             # type: ignore
            self.root.dnd_bind('<<Drop>>', self._on_dnd_drop)       # type: ignore
        except Exception:
            pass

    # ── Header ───────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg="#1a1a1a", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="✍  Nhận dạng chữ số viết tay",
                 font=self.f_title, bg="#1a1a1a", fg="white").pack()
        tk.Label(hdr, text="Vẽ hoặc tải ảnh chữ số (0–9) lên",
                 font=self.f_small, bg="#1a1a1a", fg="#aaa").pack()

    # ── Content ──────────────────────────────────────────────
    def _build_content(self):
        self._content = tk.Frame(self.root, bg="#f5f5f5", padx=16, pady=14)
        self._content.pack()

        self._build_left_panel()
        self._build_right_panel()

    # ── Panel trái (canvas + tab) ─────────────────────────────
    def _build_left_panel(self):
        self._left = tk.Frame(self._content, bg="#f5f5f5")
        self._left.grid(row=0, column=0, padx=(0, 16), sticky="n")

        # ── Tab switcher ─────────────────────────────────────
        tab_bar = tk.Frame(self._left, bg="#f5f5f5")
        tab_bar.pack(fill=tk.X, pady=(0, 6))

        self._btn_tab_draw = tk.Button(
            tab_bar, text="✍  Vẽ tay", font=self.f_label,
            relief="flat", padx=14, pady=5,
            command=lambda: self._switch_tab("draw"))
        self._btn_tab_draw.pack(side=tk.LEFT)

        self._btn_tab_upload = tk.Button(
            tab_bar, text="📁  Upload ảnh", font=self.f_label,
            relief="flat", padx=14, pady=5,
            command=lambda: self._switch_tab("upload"))
        self._btn_tab_upload.pack(side=tk.LEFT, padx=(4, 0))

        # ── Notebook frame (chứa cả hai tab) ─────────────────
        self._notebook = tk.Frame(self._left, bg="#f5f5f5")
        self._notebook.pack()

        self._build_draw_tab()
        self._build_upload_tab()
        self._switch_tab("draw")

    # ── Tab VẼ ───────────────────────────────────────────────
    def _build_draw_tab(self):
        self._draw_frame = tk.Frame(self._notebook, bg="#f5f5f5")

        tk.Label(self._draw_frame,
                 text="💡 Vẽ to, rõ, ở giữa ô — nét càng tự nhiên càng chính xác",
                 font=self.f_hint, bg="#f5f5f5", fg="#888").pack(
                     anchor="w", pady=(0, 4))

        border = tk.Frame(self._draw_frame, bg="#333", padx=2, pady=2)
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

        # Nút Xóa / Nhận dạng / Accuracy
        bf = tk.Frame(self._draw_frame, bg="#f5f5f5")
        bf.pack(fill=tk.X, pady=(8, 0))
        tk.Button(bf, text="Xóa", font=self.f_label,
                  bg="white", fg="#333", relief="flat", bd=1, padx=14, pady=6,
                  command=self.clear).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(bf, text="Nhận dạng", font=self.f_label,
                  bg="#1a1a1a", fg="white", relief="flat", padx=14, pady=6,
                  command=self.predict_draw).pack(
                      side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        tk.Button(bf, text="📊 Accuracy", font=self.f_label,
                  bg="#1565c0", fg="white", relief="flat", padx=10, pady=6,
                  command=self.show_accuracy).pack(
                      side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        af = tk.Frame(self._draw_frame, bg="#f5f5f5")
        af.pack(fill=tk.X, pady=(6, 0))
        tk.Checkbutton(af,
                       text=f"Tự động nhận dạng sau {AUTO_PREDICT_DELAY}ms",
                       variable=self.auto_predict_var,
                       font=self.f_hint, bg="#f5f5f5", fg="#666",
                       activebackground="#f5f5f5").pack(anchor="w")

    # ── Tab UPLOAD ────────────────────────────────────────────
    def _build_upload_tab(self):
        self._upload_frame = tk.Frame(self._notebook, bg="#f5f5f5")

        # Drop zone
        dz_outer = tk.Frame(self._upload_frame, bg="#333", padx=2, pady=2)
        dz_outer.pack()

        self._drop_zone = tk.Canvas(
            dz_outer,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="#2a2a2a", cursor="hand2",
            highlightthickness=0)
        self._drop_zone.pack()
        self._drop_zone.bind("<Button-1>", lambda e: self.open_file_dialog())
        self._draw_drop_hint()

        # Nút chọn file / Nhận dạng / Accuracy / Xóa
        bf2 = tk.Frame(self._upload_frame, bg="#f5f5f5")
        bf2.pack(fill=tk.X, pady=(8, 0))

        tk.Button(bf2, text="📂 Chọn ảnh", font=self.f_label,
                  bg="white", fg="#333", relief="flat", bd=1, padx=12, pady=6,
                  command=self.open_file_dialog).pack(
                      side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(bf2, text="Nhận dạng", font=self.f_label,
                  bg="#1a1a1a", fg="white", relief="flat", padx=12, pady=6,
                  command=self.predict_upload).pack(
                      side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        tk.Button(bf2, text="📊 Accuracy", font=self.f_label,
                  bg="#1565c0", fg="white", relief="flat", padx=10, pady=6,
                  command=self.show_accuracy).pack(
                      side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # Thanh info ảnh + preview 28×28
        info_frame = tk.Frame(self._upload_frame, bg="#f5f5f5")
        info_frame.pack(fill=tk.X, pady=(8, 0))

        # Cột trái: info text
        info_left = tk.Frame(info_frame, bg="#f5f5f5")
        info_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._lbl_filename = tk.Label(
            info_left, text="Chưa chọn ảnh",
            font=self.f_hint, bg="#f5f5f5", fg="#888",
            anchor="w", wraplength=200)
        self._lbl_filename.pack(anchor="w")

        self._lbl_imgsize = tk.Label(
            info_left, text="",
            font=self.f_hint, bg="#f5f5f5", fg="#aaa", anchor="w")
        self._lbl_imgsize.pack(anchor="w")

        self._lbl_invert = tk.Label(
            info_left, text="",
            font=self.f_hint, bg="#f5f5f5", fg="#888", anchor="w")
        self._lbl_invert.pack(anchor="w")

        # Cột phải: preview 28×28
        preview_box = tk.Frame(info_frame, bg="#f5f5f5")
        preview_box.pack(side=tk.RIGHT)

        tk.Label(preview_box, text="28×28 input",
                 font=self.f_hint, bg="#f5f5f5", fg="#aaa").pack()

        preview_border = tk.Frame(preview_box, bg="#555", padx=1, pady=1)
        preview_border.pack()
        self._preview_label = tk.Label(
            preview_border, bg="#1a1a1a",
            width=84, height=84)
        self._preview_label.pack()

        # Lưu ảnh upload hiện tại
        self._uploaded_pil = None

    def _draw_drop_hint(self):
        """Vẽ giao diện placeholder cho drop zone."""
        c  = self._drop_zone
        cx = CANVAS_SIZE // 2
        cy = CANVAS_SIZE // 2

        c.delete("all")
        # Viền nét đứt
        dash = 8
        for x in range(20, CANVAS_SIZE - 20, dash * 2):
            c.create_line(x, 20, min(x + dash, CANVAS_SIZE-20), 20,
                          fill="#555", width=1)
            c.create_line(x, CANVAS_SIZE-20, min(x+dash, CANVAS_SIZE-20),
                          CANVAS_SIZE-20, fill="#555", width=1)
        for y in range(20, CANVAS_SIZE - 20, dash * 2):
            c.create_line(20, y, 20, min(y + dash, CANVAS_SIZE-20),
                          fill="#555", width=1)
            c.create_line(CANVAS_SIZE-20, y, CANVAS_SIZE-20,
                          min(y+dash, CANVAS_SIZE-20), fill="#555", width=1)

        # Icon upload (mũi tên lên + đường ngang)
        c.create_polygon(cx, cy-50, cx-22, cy-28, cx-10, cy-28,
                         cx-10, cy-10, cx+10, cy-10,
                         cx+10, cy-28, cx+22, cy-28,
                         fill="#555", outline="")
        c.create_rectangle(cx-28, cy-8, cx+28, cy, fill="#555", outline="")

        c.create_text(cx, cy+26,
                      text="Nhấn để chọn ảnh",
                      font=self.f_label, fill="#aaa")
        c.create_text(cx, cy+50,
                      text="PNG · JPG · BMP · WEBP · v.v.",
                      font=self.f_hint, fill="#666")

    # ── Chuyển tab ───────────────────────────────────────────
    def _switch_tab(self, tab):
        self._current_tab = tab
        self._cancel_auto_predict()

        # Ẩn cả hai
        self._draw_frame.pack_forget()
        self._upload_frame.pack_forget()

        # Style tab button
        active_style   = dict(bg="#1a1a1a", fg="white",   relief="flat")
        inactive_style = dict(bg="#e8e8e8", fg="#555",    relief="flat")

        if tab == "draw":
            self._draw_frame.pack()
            self._btn_tab_draw.config(**active_style)
            self._btn_tab_upload.config(**inactive_style)
            self.status.config(text="Chế độ vẽ tay — sẵn sàng")
        else:
            self._upload_frame.pack()
            self._btn_tab_draw.config(**inactive_style)
            self._btn_tab_upload.config(**active_style)
            self.status.config(text="Chế độ upload — nhấn 'Chọn ảnh' hoặc kéo-thả file")

    # ── Panel kết quả (phải) ──────────────────────────────────
    def _build_right_panel(self):
        right = tk.Frame(self._content, bg="white", bd=0,
                         highlightthickness=1, highlightbackground="#ddd",
                         width=200, pady=12, padx=16)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)

        tk.Label(right, text="Kết quả", font=self.f_label,
                 bg="white", fg="#666").pack(anchor="w")

        self.result_digit = tk.Label(right, text="?",
                                     font=self.f_result,
                                     bg="white", fg="#1a1a1a")
        self.result_digit.pack(pady=(8, 0))

        self.result_conf = tk.Label(right, text="Vẽ hoặc tải ảnh",
                                    font=self.f_small, bg="white", fg="#888")
        self.result_conf.pack()

        self.result_source = tk.Label(right, text="",
                                      font=self.f_hint, bg="white", fg="#aaa")
        self.result_source.pack()

        tk.Label(right, text="Xác suất từng chữ số",
                 font=self.f_small, bg="white", fg="#999").pack(
                     anchor="w", pady=(14, 4))

        self.bars       = {}
        self.bar_labels = {}
        for d in range(10):
            rf = tk.Frame(right, bg="white")
            rf.pack(fill=tk.X, pady=1)
            tk.Label(rf, text=str(d), font=self.f_small,
                     bg="white", fg="#555", width=2, anchor="e").pack(side=tk.LEFT)
            track = tk.Frame(rf, bg="#eee", height=10)
            track.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            track.pack_propagate(False)
            bar = tk.Frame(track, bg="#bbb", height=10)
            bar.place(x=0, y=0, relwidth=0, height=10)
            lbl = tk.Label(rf, text="0%", font=self.f_small,
                           bg="white", fg="#999", width=4, anchor="w")
            lbl.pack(side=tk.LEFT)
            self.bars[d]       = (track, bar)
            self.bar_labels[d] = lbl

    # ── Status bar ────────────────────────────────────────────
    def _build_statusbar(self):
        scipy_ok = HAS_SCIPY
        self.status = tk.Label(
            self.root,
            text="Sẵn sàng | " + ("✅ scipy" if scipy_ok else "⚠️  pip install scipy"),
            font=self.f_hint, bg="#eee",
            fg="#2e7d32" if scipy_ok else "#e65100",
            anchor="w", padx=10, pady=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Vẽ tay ───────────────────────────────────────────────
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
        r = BRUSH_RADIUS
        dx = e.x - self.last_x; dy = e.y - self.last_y
        steps = max(1, int((dx**2 + dy**2)**0.5 / (r * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            ix = int(self.last_x + dx * t); iy = int(self.last_y + dy * t)
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
        self._auto_predict_job = self.root.after(
            AUTO_PREDICT_DELAY, self.predict_draw)

    def _cancel_auto_predict(self):
        if self._auto_predict_job is not None:
            self.root.after_cancel(self._auto_predict_job)
            self._auto_predict_job = None

    def clear(self):
        self._cancel_auto_predict()
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=BG_COLOR)
        self.result_digit.config(text="?", fg="#1a1a1a")
        self.result_conf.config(text="Vẽ hoặc tải ảnh", fg="#888")
        self.result_source.config(text="")
        self._reset_bars()
        self.status.config(text="Đã xóa — sẵn sàng vẽ lại")

    # ── Upload ───────────────────────────────────────────────
    def open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh chữ số",
            filetypes=UPLOAD_FILETYPES)
        if path:
            self._load_image(path)

    def _on_dnd_drop(self, event):
        """Xử lý kéo-thả file (cần tkinterdnd2)."""
        path = event.data.strip().strip("{}")
        if os.path.isfile(path):
            self._switch_tab("upload")
            self._load_image(path)

    def _load_image(self, path):
        """Tải ảnh từ path, hiển thị lên drop zone, tự nhận dạng."""
        try:
            pil = Image.open(path).convert("RGBA")
        except Exception as e:
            self.status.config(text=f"❌ Không mở được ảnh: {e}")
            return

        self._uploaded_pil = pil

        # Hiển thị ảnh gốc lên drop zone (fit vào CANVAS_SIZE × CANVAS_SIZE)
        display = pil.copy()
        display.thumbnail((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)

        # Đặt lên nền tối
        bg_img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "#2a2a2a")
        offset_x = (CANVAS_SIZE - display.width)  // 2
        offset_y = (CANVAS_SIZE - display.height) // 2
        bg_img.paste(display.convert("RGB"),
                     (offset_x, offset_y),
                     display.split()[3] if display.mode == "RGBA" else None)

        self._upload_photo = ImageTk.PhotoImage(bg_img)
        self._drop_zone.delete("all")
        self._drop_zone.create_image(0, 0, anchor="nw",
                                     image=self._upload_photo)

        # Thông tin file
        fname = os.path.basename(path)
        w, h  = pil.size
        self._lbl_filename.config(text=f"📄 {fname}", fg="#444")
        self._lbl_imgsize.config(text=f"Kích thước: {w} × {h} px")

        self.status.config(text=f"Đã tải: {fname} — đang nhận dạng...")
        self.root.update()

        # Nhận dạng ngay
        self.predict_upload()

    def predict_upload(self):
        """Nhận dạng ảnh đang được upload."""
        if self._uploaded_pil is None:
            self.status.config(
                text="⚠️  Chưa có ảnh — nhấn 'Chọn ảnh' để tải lên")
            return

        vec, arr28, inverted = preprocess_upload(self._uploaded_pil)
        if vec is None:
            self.status.config(
                text="⚠️  Không tìm thấy chữ số trong ảnh — thử ảnh khác")
            return

        # Hiển thị preview 28×28
        if arr28 is not None:
            self._preview_photo = arr28_to_photoimage(arr28, size=84)
            self._preview_label.config(image=self._preview_photo)

        # Thông tin đảo màu
        inv_text = "🔄 Đã đảo màu (nền sáng → tối)" if inverted else "✅ Giữ nguyên màu"
        self._lbl_invert.config(text=inv_text,
                                fg="#e65100" if inverted else "#2e7d32")

        self._run_prediction(vec, source="upload")

    # ── Dự đoán chung ────────────────────────────────────────
    def predict_draw(self):
        self._cancel_auto_predict()
        X = preprocess(self.image)
        if X is None:
            self.status.config(text="⚠️  Canvas trống — hãy vẽ một chữ số!")
            return
        self._run_prediction(X, source="draw")

    def _run_prediction(self, X, source="draw"):
        self.status.config(text="Đang nhận dạng...")
        self.root.update()

        pred   = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        conf   = float(np.max(probas))

        conf_color = ("#2e7d32" if conf >= 0.80
                      else "#e65100" if conf >= 0.60
                      else "#c62828")

        self.result_digit.config(text=str(pred), fg="#1a1a1a")
        self.result_conf.config(text=f"Độ tin cậy: {conf:.1%}", fg=conf_color)
        self.result_source.config(
            text="(từ ảnh upload)" if source == "upload" else "(từ vẽ tay)")

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

    def _reset_bars(self):
        for d in range(10):
            _, bar = self.bars[d]
            bar.place_configure(relwidth=0)
            bar.config(bg="#bbb")
            self.bar_labels[d].config(text="0%", fg="#999")

    # ── Accuracy window ───────────────────────────────────────
    def show_accuracy(self):
        AccuracyWindow(self.root)


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