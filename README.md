# Đề Tài 18: Nhận Dạng Chữ Số Viết Tay
**Môn học:** Trí Tuệ Nhân Tạo  
**Kỹ thuật:** Logistic Regression + Bộ dữ liệu MNIST

---

## 1. Mô Tả Đề Tài

Xây dựng hệ thống nhận dạng chữ số viết tay từ 0 đến 9. Người dùng vẽ chữ số lên giao diện, mô hình sẽ nhận diện và trả về kết quả kèm độ tin cậy.

---

## 2. Công Nghệ & Thư Viện

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.8+ |
| Mô hình | Logistic Regression (scikit-learn) |
| Dữ liệu | MNIST (70,000 ảnh 28×28) |
| Giao diện | Tkinter + Pillow |
| Lưu mô hình | joblib |

---

## 3. Cấu Trúc Dự Án

```
digit_recognition/
├── train_model.py      # Huấn luyện và lưu mô hình
├── evaluate_model.py   # Đánh giá chi tiết + biểu đồ
├── draw_predict.py     # Ứng dụng giao diện vẽ & dự đoán
├── digit_model.pkl     # File mô hình đã huấn luyện
└── README.md
```



## 4. Phương Pháp

### 4.1 Bộ Dữ Liệu MNIST

- **70,000 ảnh** grayscale kích thước 28×28 pixel
- **10 lớp** (chữ số 0–9), mỗi lớp ~7,000 mẫu
- Chia: 80% huấn luyện (56,000) / 20% kiểm tra (14,000)
- Chuẩn hóa pixel: [0, 255] → [0.0, 1.0]

### 4.2 Mô Hình Logistic Regression

Logistic Regression là thuật toán phân loại tuyến tính sử dụng hàm sigmoid (hoặc softmax cho đa lớp) để ánh xạ đầu vào thành xác suất:

```
P(y = k | x) = softmax(Wₖᵀx + bₖ)
```

Với bài toán 10 lớp, dùng chiến lược **One-vs-Rest (OvR)**: huấn luyện 10 bộ phân loại nhị phân, mỗi bộ phân biệt một chữ số với 9 chữ số còn lại.

**Tham số:**
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `solver` | `saga` | Phù hợp bộ dữ liệu lớn, hỗ trợ L1/L2 |
| `C` | `0.1` | Regularization (tránh overfitting) |
| `max_iter` | `1000` | Số vòng lặp tối đa |
| `multi_class` | `ovr` | One-vs-Rest |

### 4.3 Tiền Xử Lý Ảnh Người Dùng Vẽ

Để ảnh vẽ tay tương thích với MNIST:

1. **Đảo màu**: MNIST có nét trắng trên nền đen → đảo ảnh vẽ
2. **Threshold**: Loại bỏ nhiễu nhỏ (< 50)
3. **Crop**: Cắt vùng chứa chữ số (bounding box)
4. **Padding**: Thêm viền để tạo hình vuông
5. **Resize**: Về 28×28 pixel (LANCZOS)
6. **Normalize**: Chia 255.0

## 5. Kết Quả

| Chỉ số | Giá trị |
|---|---|
| Độ chính xác (test set) | ~92% |
| Thời gian huấn luyện | ~3–8 phút |
| Thời gian dự đoán | < 1ms/ảnh |

## 6. Hướng Dẫn Chạy

### Bước 1: Cài đặt thư viện
pip install scikit-learn numpy matplotlib pillow joblib seaborn

Bước 2: Huấn luyện mô hình
python train_model.py
→ Tạo ra file `digit_model.pkl`

### Bước 3: Đánh giá mô hình (tuỳ chọn)
```bash
python evaluate_model.py
```

### Bước 4: Chạy ứng dụng vẽ
```bash
python draw_predict.py
```

---

## 7. Ưu & Nhược Điểm

**Ưu điểm:**
- Đơn giản, dễ hiểu, phù hợp làm nền tảng học ML
- Huấn luyện nhanh so với Deep Learning
- Có thể giải thích được (trọng số tuyến tính)

**Nhược điểm:**
- Không nắm bắt được đặc trưng cục bộ (spatial features) như CNN
- Nhạy với sự khác biệt về nét vẽ (độ dày, góc nghiêng)
- Độ chính xác giới hạn ~92% (CNN đạt >99%)

**Hướng cải tiến:**
- Dùng SVM với kernel RBF (≈ 98%)
- Áp dụng CNN (≈ 99.7%)
- Data augmentation để cải thiện robustness
