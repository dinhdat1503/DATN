# Giải Thích Chi Tiết Mã Nguồn Tiền Xử Lý Ảnh (preprocess_enhance.py)

Tài liệu này giải thích cặn kẽ từng dòng code quan trọng trong tệp `scripts/preprocess_enhance.py` để bạn hiểu rõ bản chất toán học và thuật toán đứng sau các bước tiền xử lý.

---

## 1. Hàm `ben_graham_normalization` (Chuẩn hóa màu Ben Graham)

Hàm này được dùng để loại bỏ sự chênh lệch ánh sáng do thiết bị (Device Bias) bằng cách làm phẳng ánh sáng nền.

```python
def ben_graham_normalization(img: np.ndarray, sigma_ratio: float = 1 / 6, scale: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    # 1. Tính toán tham số làm mờ (sigma) dựa trên kích thước ảnh
    sigma = int(max(h, w) * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1  # Kích thước kernel của OpenCV bắt buộc phải là số lẻ

    # 2. Tạo một bức ảnh "chỉ chứa nền ánh sáng" bằng thuật toán làm mờ Gaussian
    # Việc làm mờ với sigma lớn sẽ xóa hết chi tiết mạch máu, chỉ giữ lại độ sáng nền cục bộ.
    local_avg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)

    # 3. Trừ đi nền ánh sáng và tịnh tiến lên mức trung bình (scale = 128)
    # img - local_avg: Loại bỏ hoàn toàn vùng quá sáng/quá tối.
    # + scale: Kéo màu sắc về mức xám trung tính (128).
    result = img.astype(np.float32) - local_avg + scale

    # 4. Giới hạn giá trị pixel từ 0 đến 255 để tránh bị lỗi hiển thị
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
```

**Nguyên lý hoạt động:** Thay vì cố gắng tăng sáng những chỗ tối, thuật toán này lấy ảnh gốc trừ đi "phiên bản bị làm mờ" của chính nó. Phiên bản làm mờ đại diện cho sự phân bố ánh sáng (chỗ nào có đèn flash, chỗ nào bị râm). Khi trừ đi, ta triệt tiêu được sự chênh lệch ánh sáng đó.

---

## 2. Hàm `apply_clahe` (Tăng cường độ tương phản cục bộ)

Hàm này làm nổi bật các chi tiết nhỏ như vi mạch máu, vết nứt, tổn thương trên võng mạc.

```python
def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    # 1. Chuyển đổi từ không gian màu mặc định (BGR) sang không gian màu LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 2. Tách thành 3 rãnh (kênh) riêng biệt: L, A, B
    # Kênh L (Luminance): Chứa thông tin về độ sáng (sáng/tối).
    # Kênh A, B: Chứa thông tin về màu sắc (xanh/đỏ/vàng...).
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 3. Khởi tạo đối tượng CLAHE
    # clipLimit=2.0: Giới hạn độ khuếch đại, giúp không bị nhiễu hạt (noise) quá mức.
    # tileGridSize=(8,8): Chia ảnh thành lưới 8x8 ô vuông để xử lý tương phản TỪNG Ô MỘT, 
    # giúp làm rõ cả những vùng tối nhất và sáng nhất trong cùng một bức ảnh.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # 4. Áp dụng CLAHE CHỈ LÊN KÊNH L (độ sáng)
    # Việc này cực kỳ quan trọng: Giúp làm sắc nét chi tiết mà KHÔNG LÀM SAI LỆCH màu sắc gốc của tổn thương.
    l_enhanced = clahe.apply(l_channel)

    # 5. Ghép rãnh L đã được tăng cường lại với rãnh A, B ban đầu
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # 6. Chuyển ngược lại về hệ màu BGR để lưu trữ
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result
```

---

## 3. Khối xử lý đa luồng (ProcessPoolExecutor) trong hàm `process_all_images`

Để xử lý bộ dữ liệu hàng nghìn ảnh, việc chạy tuần tự (từng ảnh một) sẽ cực kỳ chậm. Code của bạn đã áp dụng kỹ thuật chạy song song:

```python
# Mở một "hồ chứa" các bộ vi xử lý (CPU Cores), tối đa = số lượng workers
with ProcessPoolExecutor(max_workers=workers) as executor:
    for img_path in image_files:
        # Giao phó nhiệm vụ enhance_single_image cho các CPU chạy độc lập cùng lúc
        future = executor.submit(enhance_single_image, ...)
        futures[future] = img_path.name
```
*   **Ý nghĩa:** Ví dụ máy bạn có 4 nhân CPU, thay vì 1 nhân làm việc mệt mỏi với 5000 ảnh, đoạn code này chia đều cho cả 4 nhân cùng làm việc đồng thời. Tốc độ tiền xử lý dữ liệu sẽ nhanh gấp nhiều lần (gần như tỷ lệ thuận với số nhân CPU). 

**Kết luận:** Bạn có thể dùng trực tiếp file giải thích này khi giáo viên hướng dẫn yêu cầu giải thích "Tại sao em lại code như vậy?" hoặc "Bản chất toán học của CLAHE và Ben Graham trong đồ án của em là gì?".
