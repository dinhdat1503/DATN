# HƯỚNG DẪN CHẠY DEBUG CHI TIẾT TỪNG DÒNG CODE CỦA HÀM HUẤN LUYỆN `run_epoch()`
## DỰ ÁN: ODIR-5K MULTI-TASK LEARNING

Tài liệu này được biên soạn theo phong cách **Chạy từng dòng lệnh (Line-by-Line Debugger Walkthrough)**. Chúng ta sẽ đóng vai trò là trình gỡ lỗi (Debugger), đi qua **từng dòng code một từ dòng đầu tiên đến dòng cuối cùng** của hàm huấn luyện chính `run_epoch()`. Tài liệu này giải thích ý nghĩa của từng biến số được khởi tạo, kiểu dữ liệu của chúng thay đổi ra sao và luồng chạy của chương trình chuyển động như thế nào.

---

## 1. Định Vị Toàn Bộ Hàm `run_epoch()` Trong File `train.py`

Hàm `run_epoch()` nằm từ dòng **245** đến dòng **341** trong tệp [train.py](file:///media/dinhdat/OD/DOANTOTNGHIEP/DOANTOTNGHIEP/train.py). Dưới đây là toàn bộ mã nguồn của hàm này:

```python
def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mode: str,
    age_mean: float,
    age_std: float,
    threshold: float | list[float] | torch.Tensor = 0.5,
) -> dict:
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    # Accumulators
    total_loss = total_cls = total_reg = 0.0
    all_probs    = []   # [N, 8] sigmoid probabilities
    all_targets  = []   # [N, 8] ground truth labels
    all_age_pred = []   # [N] predicted age (normalized)
    all_age_true = []   # [N] true age (normalized)

    it = enumerate(loader)
    if HAS_TQDM:
        it = enumerate(tqdm(loader, desc=f"  {mode}", leave=False, ncols=90))

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for batch_idx, batch in it:
            images   = batch["image"].to(device)
            labels   = batch["labels"].to(device)
            age_true = batch["age"].to(device)

            # Forward
            output   = model(images)
            logits   = output["logits"]
            age_pred = output["age_pred"]

            # Loss
            loss, detail = criterion(logits, labels, age_pred, age_true)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accumulate
            bs = images.size(0)
            total_loss += detail["loss_total"] * bs
            total_cls  += detail["loss_cls"]   * bs
            total_reg  += detail["loss_reg"]   * bs

            probs = torch.sigmoid(logits).detach().cpu().tolist()
            all_probs.extend(probs)
            all_targets.extend(labels.detach().cpu().tolist())
            all_age_pred.extend(age_pred.squeeze(1).detach().cpu().tolist())
            all_age_true.extend(age_true.squeeze(1).detach().cpu().tolist())

    n = len(loader.dataset)
    metrics = {
        f"{mode}_loss":     total_loss / n,
        f"{mode}_loss_cls": total_cls  / n,
        f"{mode}_loss_reg": total_reg  / n,
    }

    # F1-macro (threshold 0.5)
    import torch as _t
    probs_t   = _t.FloatTensor(all_probs)
    targets_t = _t.FloatTensor(all_targets)
    ml_metrics = compute_multilabel_metrics(probs_t, targets_t, threshold=threshold)
    for k, v in ml_metrics.items():
        metrics[f"{mode}_{k}"] = v

    # AUC-ROC
    metrics[f"{mode}_auc_roc"] = compute_auc_roc(all_probs, all_targets)

    # Age metrics
    age_m = compute_mae_pearson(all_age_pred, all_age_true, age_mean, age_std)
    metrics[f"{mode}_age_mae"]     = age_m["mae"]
    metrics[f"{mode}_age_pearson"] = age_m["pearson"]

    return metrics
```

---

## 2. HƯỚNG DẪN CHẠY DEBUG TỪNG DÒNG LỆNH (STEP-BY-STEP WALKTHROUGH)

Hãy tưởng tượng bạn đang nhấn nút **Step Over (F10)** trên trình gỡ lỗi của PyCharm để chạy qua từng dòng code. Dưới đây là những gì CPU/GPU thực thi tại mỗi bước:

---

### Bước 2.1: Khởi tạo và thiết lập trạng thái ban đầu

*   **DÒNG 1: `def run_epoch(...)`**
    *   *Debugger thực thi:* Nhận các đối số đầu vào được truyền từ hàm `train()` chính.
    *   *Trạng thái biến:* Nạp các đối tượng PyTorch vào bộ nhớ gồm: `model` (mô hình CNN/Swin đã dựng), `loader` (bộ nạp dữ liệu), `criterion` (hàm tính loss), `optimizer` (bộ tối ưu hóa gradient), và các siêu tham số `device` (GPU), `mode` ("train"/"val").
*   **DÒNG 10: `is_train = (mode == "train")`**
    *   *Debugger thực thi:* Thực hiện phép so sánh chuỗi ký tự. Nếu `mode` là `"train"`, biến logic `is_train` nhận giá trị **`True`**. Ngược lại, nếu ở chế độ validation hoặc test, `is_train` nhận giá trị **`False`**.
*   **DÒNG 11: `model.train() if is_train else model.eval()`**
    *   *Debugger thực thi:* Gọi hàm thiết lập trạng thái của mô hình PyTorch.
        *   Nếu `is_train` là `True`: Kích hoạt chế độ huấn luyện. PyTorch sẽ bật các tầng điều hòa như Dropout và BatchNorm.
        *   Nếu `is_train` là `False`: Kích hoạt chế độ đánh giá. PyTorch sẽ khóa chặt các neuron Dropout và BatchNorm lại để mô hình suy luận ổn định.
*   **DÒNG 14: `total_loss = total_cls = total_reg = 0.0`**
    *   *Debugger thực thi:* Khởi tạo 3 biến số thực (float) trên RAM có giá trị ban đầu là `0.0`. Các biến này đóng vai trò là các bộ tích lũy (accumulators) dùng để cộng dồn điểm phạt loss của toàn bộ các batch trong epoch.
*   **DÒNG 15 đến 18: Khởi tạo các danh sách chứa kết quả (Empty Lists)**
    *   *Debugger thực thi:* Khởi tạo các mảng rỗng trên RAM: `all_probs` (chứa xác suất dự đoán bệnh), `all_targets` (chứa nhãn gốc), `all_age_pred` (chứa tuổi dự đoán), `all_age_true` (chứa tuổi thực tế). Các mảng này sẽ gom toàn bộ kết quả của 5,600 ảnh võng mạc sau khi duyệt qua hết các batch để tính toán chỉ số F1-macro và MAE cuối epoch.
*   **DÒNG 20: `it = enumerate(loader)`**
    *   *Debugger thực thi:* Khởi tạo một trình duyệt (iterator) kết hợp phép đếm chỉ số. Mỗi lần lặp sẽ trả về một cặp giá trị: `batch_idx` (chỉ số lô ảnh, bắt đầu từ 0, 1, 2...) và `batch` (dữ liệu của lô ảnh đó).
*   **DÒNG 21-22: `if HAS_TQDM: it = ...`**
    *   *Debugger thực thi:* Kiểm tra xem thư viện vẽ thanh tiến trình `tqdm` có được cài đặt hay không. Nếu có, bao bọc trình duyệt `it` bằng `tqdm` để hiển thị phần trăm tiến độ chạy huấn luyện ra màn hình terminal cho người dùng theo dõi.
*   **DÒNG 24: `ctx = torch.no_grad() if not is_train else torch.enable_grad()`**
    *   *Debugger thực thi:* Thiết lập ngữ cảnh gradient của PyTorch.
        *   Nếu ở chế độ huấn luyện: Kích hoạt `torch.enable_grad()` để PyTorch theo dõi và xây dựng đồ thị tính đạo hàm ngược.
        *   Nếu ở chế độ kiểm định: Kích hoạt `torch.no_grad()` để tắt bộ ghi nhớ gradient, giúp giải phóng bộ nhớ GPU VRAM và đẩy nhanh tốc độ chẩn đoán.
*   **DÒNG 26: `with ctx:`**
    *   *Debugger thực thi:* Đi vào ngữ cảnh gradient đã chọn ở trên.

---

### Bước 2.2: Vòng lặp duyệt qua từng Lô dữ liệu (Batch size = 16)

*   **DÒNG 27: `for batch_idx, batch in it:`**
    *   *Debugger thực thi:* Bắt đầu vòng lặp. `DataLoader` tiến hành tải một lô gồm 16 ảnh võng mạc đáy mắt từ ổ đĩa cứng lên bộ nhớ RAM.
*   **DÒNG 28 đến 30: Đẩy ma trận dữ liệu lên card đồ họa GPU**
    *   *Debugger thực thi:* Sao chép các ma trận số từ bộ nhớ RAM của máy tính lên RAM của card đồ họa GPU (`to(device)`) để sẵn sàng cho tính toán song song hiệu năng cao:
        *   `images`: Tensor 4 chiều kích thước $[16, 3, 384, 384]$.
        *   `labels`: Tensor 2 chiều kích thước $[16, 8]$.
        *   `age_true`: Tensor 2 chiều kích thước $[16, 1]$.
*   **DÒNG 33: `output = model(images)` (BƯỚC 1: LAN TRUYỀN TIẾN)**
    *   *Debugger thực thi:* Truyền ma trận ảnh `images` vào mô hình. PyTorch thực hiện hàng triệu phép nhân ma trận trượt nhân lọc trên ảnh thông qua tệp `efficientnet_mtl.py` hoặc `swin_mtl.py`.
    *   *Kết quả:* Trả về biến `output` dạng Python Dictionary chứa hai ma trận dự đoán thô.
*   **DÒNG 34-35: Trích xuất đầu ra dự đoán**
    *   *Debugger thực thi:* Tách biến `output` thành:
        *   `logits`: Tensor kích thước $[16, 8]$ chứa điểm số dự đoán thô của 8 bệnh.
        *   `age_pred`: Tensor kích thước $[16, 1]$ chứa tuổi võng mạc chuẩn hóa dự đoán.
*   **DÒNG 38: `loss, detail = criterion(...)` (BƯỚC 2: TÍNH ĐIỂM PHẠT LOSS)**
    *   *Debugger thực thi:* Gọi lớp `MultiTaskLoss` trong tệp `src/loss.py` để so sánh các dự đoán (`logits`, `age_pred`) với đáp án thực tế của bác sĩ (`labels`, `age_true`).
    *   *Trạng thái biến:* Trả về biến **`loss`** (một PyTorch tensor vô hướng đại diện cho điểm phạt sai số chung) và **`detail`** (một dict chứa các con số float ghi nhận điểm loss chi tiết).
*   **DÒNG 40: `if is_train:`**
    *   *Debugger thực thi:* Kiểm tra xem có phải đang ở chế độ huấn luyện hay không. Nếu đúng, chương trình đi vào khối lệnh tự sửa sai (Dòng 41-44). Nếu ở chế độ validation, chương trình bỏ qua khối này và nhảy thẳng xuống dòng 47.
*   **DÒNG 41: `optimizer.zero_grad()`**
    *   *Debugger thực thi:* Gọi bộ tối ưu AdamW thực hiện xóa sạch các vết tích lũy đạo hàm/lỗi sai cũ trong bộ nhớ gradient để chuẩn bị cho đợt cập nhật mới.
*   **DÒNG 42: `loss.backward()` (BƯỚC 3: LAN TRUYỀN NGƯỢC - DÒ TÌM LỖI SAI)**
    *   *Debugger thực thi:* PyTorch đi ngược từ điểm phạt `loss` về phía các lớp tích chập ban đầu, tính toán đạo hàm riêng cho từng tham số trọng số (Weight) của ô lưới lọc $3\times3$, xác định xem trọng số nào đã gây ra lỗi sai trong batch này.
*   **DÒNG 43: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`**
    *   *Debugger thực thi:* Kiểm tra chuẩn L2 của vector gradient vừa tính. Nếu chuẩn này vượt quá `1.0`, tự động co dãn thu nhỏ vector gradient về dải giới hạn để tránh hiện tượng bùng nổ gradient gây sụp đổ mạng.
*   **DÒNG 44: `optimizer.step()` (BƯỚC 4: CẬP NHẬT TRỌNG SỐ)**
    *   *Debugger thực thi:* Bộ tối ưu hóa AdamW trực tiếp thực hiện phép toán cộng/trừ để thay đổi các giá trị trọng số trong mạng neural dựa trên đạo hàm lỗi sai thu được, giúp các ô lưới lọc thông minh hơn.
*   **DÒNG 47: `bs = images.size(0)`**
    *   *Debugger thực thi:* Trích xuất kích thước thực tế của lô ảnh hiện tại. Thông thường là `16`, nhưng ở lô ảnh cuối cùng của tập dữ liệu, con số này có thể nhỏ hơn (Ví dụ: `8` ảnh còn dư lại).
*   **DÒNG 48 đến 50: Cộng dồn điểm loss của toàn bộ Epoch**
    *   *Debugger thực thi:* Nhân điểm loss trung bình của lô với kích thước lô `bs` và cộng dồn vào các biến tích lũy `total_loss`, `total_cls`, `total_reg`.
*   **DÒNG 52: `probs = torch.sigmoid(logits).detach().cpu().tolist()`**
    *   *Debugger thực thi:* 
        1.  Gọi hàm `torch.sigmoid` để quy đổi điểm logits thô thành xác suất mắc bệnh nằm trong khoảng $[0, 1]$.
        2.  `detach()`: Ngắt kết nối tensor khỏi đồ thị gradient của PyTorch.
        3.  `cpu()`: Sao chép ma trận xác suất này từ card đồ họa GPU về bộ nhớ RAM máy tính.
        4.  `tolist()`: Chuyển đổi tensor PyTorch thành một mảng Python dạng danh sách lồng nhau (List of Lists) để tiện xử lý toán học thông thường.
*   **DÒNG 53 đến 56: Thu thập toàn bộ kết quả của Epoch**
    *   *Debugger thực thi:* Sử dụng phép toán `extend` để nối thêm 16 kết quả dự đoán và nhãn gốc của batch hiện tại vào các danh sách tổng hợp `all_probs`, `all_targets`, `all_age_pred`, `all_age_true`.
    *   *Vòng lặp tiếp tục:* Debugger quay lại Dòng 27 để nạp batch ảnh tiếp theo, lặp đi lặp lại cho đến khi duyệt hết toàn bộ 5,600 ảnh đáy mắt võng mạc của tập Train.

---

### Bước 2.3: Tính toán chỉ số y sinh cuối cùng của Epoch

Sau khi duyệt qua toàn bộ các batch (kết thúc vòng lặp `for`), trình gỡ lỗi thoát khỏi khối `with ctx:` và thực hiện các dòng lệnh cuối cùng:

*   **DÒNG 58: `n = len(loader.dataset)`**
    *   *Debugger thực thi:* Trích xuất tổng số lượng mẫu ảnh thực tế trong tập dữ liệu (Ví dụ: `5600`).
*   **DÒNG 59 đến 63: Tính Loss trung bình của Epoch**
    *   *Debugger thực thi:* Khởi tạo Dictionary `metrics`. Lấy tổng điểm loss đã tích lũy chia cho tổng số lượng ảnh `n` để ra giá trị loss trung bình của toàn bộ Epoch:
        *   `metrics["train_loss"] = total_loss / n`
*   **DÒNG 66 đến 70: Tính điểm chẩn đoán F1-macro**
    *   *Debugger thực thi:* 
        1.  Chuyển đổi danh sách xác suất và nhãn gốc trên RAM thành PyTorch tensors `probs_t` và `targets_t`.
        2.  Gọi hàm tiện ích `compute_multilabel_metrics()` trong tệp `src/utils.py` để so sánh toàn bộ dự đoán với nhãn gốc và tính toán ra điểm số F1-macro tổng hợp.
        3.  Lưu các chỉ số F1 vào từ điển `metrics`.
*   **DÒNG 73: Tính chỉ số phân biệt bệnh lý AUC-ROC**
    *   *Debugger thực thi:* Gọi hàm `compute_auc_roc()` tính toán chỉ số diện tích dưới đường cong ROC để đo năng lực xếp hạng chẩn đoán bệnh của mô hình và lưu vào `metrics`.
*   **DÒNG 76 đến 78: Tính sai số năm tuổi y khoa (Age MAE)**
    *   *Debugger thực thi:* Gọi hàm `compute_mae_pearson()` trong tệp `src/utils.py`. Hàm này tự động denormalize tuổi võng mạc, so sánh tuổi thực và dự đoán, tính ra sai số tuyệt đối trung bình **MAE (năm tuổi)** và hệ số tương quan tuyến tính Pearson.
*   **DÒNG 80: `return metrics`**
    *   *Debugger thực thi:* Trả từ điển kết quả `metrics` hoàn chỉnh về cho tệp `train.py` để kết thúc chu kỳ huấn luyện của một epoch.

---

*Tài liệu hướng dẫn chạy debug mã nguồn huấn luyện được biên soạn bởi Antigravity nhằm hỗ trợ Ngô Đình Đạt hiện thực hóa Đồ án Tốt nghiệp xuất sắc.*
