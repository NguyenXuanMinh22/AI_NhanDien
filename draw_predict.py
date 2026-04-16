"""
================================================
  ĐỀ TÀI 18: NHẬN DẠNG CHỮ SỐ VIẾT TAY
  draw_predict.py — Ứng dụng vẽ & dự đoán
================================================
"""

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_PATH  = "digit_model.pkl"
CANVAS_SIZE = 300         # pixel của canvas vẽ
BRUSH_RADIUS = 10         # bán kính bút vẽ
BG_COLOR    = "#1a1a1a"   # màu nền canvas (tối)
FG_COLOR    = "white"     # màu nét vẽ

# ============================================================
# TẢI MÔ HÌNH
# ============================================================
print("Đang tải mô hình...")
model = joblib.load(MODEL_PATH)
print("Tải mô hình thành công!")

# ============================================================
# TIỀN XỬ LÝ ẢNH (chuẩn MNIST)
# ============================================================
def preprocess(pil_image):
    """
    Chuyển ảnh vẽ tay → vector 784 chiều theo chuẩn MNIST.
    Bước:
      1. Chuyển grayscale
      2. Đảo màu (MNIST: trắng trên đen)
      3. Áp threshold để loại nhiễu
      4. Cắt vùng chứa chữ số (bounding box)
      5. Padding về hình vuông
      6. Resize về 28×28
      7. Chuẩn hóa [0, 1]
    """
    img = np.array(pil_image.convert('L'))

    # Đảo màu: nét vẽ trắng → nét vẽ tối; nền đen → nền sáng
    img = 255 - img

    # Threshold
    img = np.where(img < 50, 0, 255).astype(np.uint8)

    # Tìm bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None  # canvas trống

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Thêm padding nhỏ quanh chữ số
    pad = 10
    y_min = max(0, y_min - pad)
    y_max = min(img.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(img.shape[1], x_max + pad)

    img = img[y_min:y_max + 1, x_min:x_max + 1]

    # Padding về hình vuông
    h, w = img.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = img

    # Resize 28×28
    pil_28 = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
    arr = np.array(pil_28, dtype=np.float32) / 255.0

    return arr.reshape(1, -1)


# ============================================================
# GIAO DIỆN TKINTER
# ============================================================
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận dạng chữ số viết tay — Đề tài 18")
        self.root.configure(bg="#f5f5f5")
        self.root.resizable(False, False)

        # Fonts
        title_font  = tkfont.Font(family="Helvetica", size=14, weight="bold")
        label_font  = tkfont.Font(family="Helvetica", size=11)
        result_font = tkfont.Font(family="Courier", size=48, weight="bold")
        small_font  = tkfont.Font(family="Helvetica", size=9)

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

        # Canvas vẽ
        left = tk.Frame(content, bg="#f5f5f5")
        left.grid(row=0, column=0, padx=(0, 16))

        tk.Label(left, text="Bảng vẽ", font=label_font,
                 bg="#f5f5f5", fg="#333").pack(anchor="w", pady=(0, 6))

        border = tk.Frame(left, bg="#333", padx=2, pady=2)
        border.pack()

        self.canvas = tk.Canvas(border, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg=BG_COLOR, cursor="pencil",
                                highlightthickness=0)
        self.canvas.pack()

        # PIL image (nền đen)
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        self.draw  = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None
        self.canvas.bind("<Button-1>",   self.on_click)
        self.canvas.bind("<B1-Motion>",  self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Nút bấm
        btn_frame = tk.Frame(left, bg="#f5f5f5")
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(btn_frame, text="Xóa", font=label_font,
                  bg="white", fg="#333", relief="flat",
                  bd=1, padx=16, pady=6,
                  command=self.clear).pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(btn_frame, text="Nhận dạng", font=label_font,
                  bg="#1a1a1a", fg="white", relief="flat",
                  padx=16, pady=6,
                  command=self.predict).pack(side=tk.RIGHT, expand=True,
                                            fill=tk.X, padx=(8, 0))

        # ── Khu vực kết quả ──────────────────────────────────
        right = tk.Frame(content, bg="white", bd=0,
                         highlightthickness=1,
                         highlightbackground="#ddd",
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
                                    font=small_font,
                                    bg="white", fg="#888")
        self.result_conf.pack()

        # Phân phối xác suất
        tk.Label(right, text="Xác suất từng chữ số",
                 font=small_font, bg="white", fg="#999").pack(
                     anchor="w", pady=(16, 4))

        self.bars = {}
        self.bar_labels = {}

        for d in range(10):
            row_f = tk.Frame(right, bg="white")
            row_f.pack(fill=tk.X, pady=1)

            tk.Label(row_f, text=str(d), font=small_font,
                     bg="white", fg="#555", width=2,
                     anchor="e").pack(side=tk.LEFT)

            track = tk.Frame(row_f, bg="#eee", height=10)
            track.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            track.pack_propagate(False)

            bar = tk.Frame(track, bg="#bbb", height=10)
            bar.place(x=0, y=0, relwidth=0, height=10)

            lbl = tk.Label(row_f, text="0%", font=small_font,
                           bg="white", fg="#999", width=4, anchor="w")
            lbl.pack(side=tk.LEFT)

            self.bars[d] = (track, bar)
            self.bar_labels[d] = lbl

        # ── Status bar ────────────────────────────────────────
        self.status = tk.Label(root, text="Sẵn sàng",
                               font=small_font, bg="#eee",
                               fg="#666", anchor="w", padx=10, pady=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Sự kiện vẽ ───────────────────────────────────────────
    def on_click(self, e):
        self.last_x, self.last_y = e.x, e.y

    def on_drag(self, e):
        if self.last_x is None:
            return
        r = BRUSH_RADIUS
        # Vẽ lên canvas Tk
        self.canvas.create_line(self.last_x, self.last_y, e.x, e.y,
                                fill=FG_COLOR, width=r * 2,
                                capstyle=tk.ROUND, smooth=True)
        # Vẽ lên PIL image
        self.draw.line([self.last_x, self.last_y, e.x, e.y],
                       fill=FG_COLOR, width=r * 2)
        self.last_x, self.last_y = e.x, e.y

    def on_release(self, e):
        self.last_x = self.last_y = None

    # ── Xóa ──────────────────────────────────────────────────
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=BG_COLOR)
        self.result_digit.config(text="?", fg="#1a1a1a")
        self.result_conf.config(text="Vẽ để bắt đầu")
        self._reset_bars()
        self.status.config(text="Đã xóa — sẵn sàng vẽ lại")

    def _reset_bars(self):
        for d in range(10):
            track, bar = self.bars[d]
            bar.place_configure(relwidth=0)
            bar.config(bg="#bbb")
            self.bar_labels[d].config(text="0%", fg="#999")

    # ── Dự đoán ──────────────────────────────────────────────
    def predict(self):
        self.status.config(text="Đang nhận dạng...")
        self.root.update()

        X = preprocess(self.image)
        if X is None:
            self.status.config(text="Canvas trống — hãy vẽ một chữ số!")
            return

        pred    = model.predict(X)[0]
        probas  = model.predict_proba(X)[0]
        conf    = np.max(probas)

        # Cập nhật kết quả
        self.result_digit.config(text=str(pred), fg="#1a1a1a")
        self.result_conf.config(
            text=f"Độ tin cậy: {conf:.1%}",
            fg="#2e7d32" if conf >= 0.8 else "#e65100"
        )

        # Cập nhật thanh xác suất
        self._reset_bars()
        for d in range(10):
            p = probas[d]
            track, bar = self.bars[d]
            bar.config(bg="#1a1a1a" if d == pred else "#90caf9")
            bar.place_configure(relwidth=float(p))
            self.bar_labels[d].config(
                text=f"{p:.0%}",
                fg="#1a1a1a" if d == pred else "#666"
            )

        self.status.config(
            text=f"Kết quả: chữ số {pred} | Tin cậy: {conf:.2%}"
        )


# ============================================================
# CHẠY ỨNG DỤNG
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = DigitApp(root)
    root.mainloop()