# HƯỚNG DẪN GIẢI THÍCH CHI TIẾT HÀM HUẤN LUYỆN TRONG NOTEBOOK (`run_experiment`)
## DỰ ÁN: ODIR-5K MULTI-TASK LEARNING

Tài liệu này cung cấp phần giải thích cấu trúc thuật toán và phân tích chi tiết từng khối lệnh quan trọng trong hàm huấn luyện chính **`run_experiment()`** ở **CELL 8** của các notebook chạy trên Kaggle (`odir5k_cnn_kaggle.ipynb` và `odir5k_swin_kaggle.ipynb`). Nội dung được biên soạn chuẩn cấu trúc học thuật nhằm hỗ trợ trực tiếp việc viết chương **"Hiện thực hóa hệ thống và Thử nghiệm lâm sàng"** của Đồ án Tốt nghiệp xuất sắc.

---

## 1. Tổng Quan Cấu Trúc Hàm Huấn Luyện `run_experiment`

Hàm `run_experiment()` đóng vai trò là **bộ tích hợp toàn diện (All-in-One Orchestrator)** của notebook. Nó đóng gói toàn bộ quy trình từ cấu hình siêu tham số, khởi tạo Loader, huấn luyện đa nhiệm qua từng epoch, dừng sớm chống quá khớp, quét tìm ngưỡng động tối ưu, đến đánh giá kiểm thử đối chứng trên tập TEST và kết xuất kết quả ra file JSON.

```python
def run_experiment(config_path, img_dir_override=None):
```
*   **`config_path`**: Đường dẫn đến tệp YAML chứa tất cả các siêu tham số huấn luyện (như `epochs`, `lr`, `batch_size`, `use_weighted_sampler`, v.v.).
*   **`img_dir_override`**: Cho phép ghi đè thư mục ảnh (ví dụ: ép mô hình dùng ảnh thô `RAW_DIR` ở EXP 1 hoặc ảnh tiền xử lý `ENH_DIR` ở EXP 2, 3).

---

## 2. Phân Tích Chi Tiết Từng Khối Lệnh Quan Trọng

### 2.1. Thiết lập Data Dataloader và Cân Bằng Lớp (WRS)
Khối mã này chịu trách nhiệm khởi tạo tập dữ liệu võng mạc đáy mắt và cấu hình bộ lấy mẫu cân bằng lớp `WeightedRandomSampler` cùng các kỹ thuật trộn ảnh MixUp/CutMix:

```python
        # 1. Khởi tạo tập dữ liệu Train, Validation và Test
        ds_train = mk_ds('train', tf_train)
        ds_val   = mk_ds('val',   tf_val)
        ds_test  = mk_ds('test',  tf_val)

        # 2. Cấu hình WeightedRandomSampler (WRS) nếu được bật trong tệp cấu hình YAML
        if use_wrs:
            sampler = make_weighted_sampler(ds_train)
            base_kw = dict(batch_size=batch, sampler=sampler,
                           num_workers=2, pin_memory=True, drop_last=True)
        else:
            base_kw = dict(batch_size=batch, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
```
*   **WeightedRandomSampler (WRS):** Khi bật `use_weighted_sampler: true` (ở EXP 3 và EXP 6), loader sẽ tự động tắt `shuffle=True` và nạp `sampler=sampler`. Bộ lấy mẫu này sẽ dựa trên trọng số nghịch đảo tần suất xuất hiện của các bệnh y khoa để ưu tiên nạp các ca bệnh hiếm (như Hypertension, AMD) vào GPU.

### 2.2. Kích Hoạt MixUp và CutMix Xen Kẽ
Đối với kịch bản có tăng cường dữ liệu nâng cao (EXP 3 và EXP 6), khối lệnh này định nghĩa một lớp con chuyên biệt để trộn ảnh ngẫu nhiên cho từng batch:

```python
        # Định nghĩa bộ trộn ảnh ngẫu nhiên 50% MixUp và 50% CutMix
        class _MixCut:
            def __init__(self):
                self.mx = MixUpCollator(alpha=aug_cfg.get('mixup_alpha',0.4), prob=1.0)
                self.cx = CutMixCollator(alpha=aug_cfg.get('cutmix_alpha',1.0), prob=1.0)
            def __call__(self, b): 
                return self.mx(b) if random.random() < 0.5 else self.cx(b)

        # Gán collate_fn vào DataLoader để tự động trộn ảnh trước khi đưa lên GPU
        if use_mixup and use_cutmix:
            train_loader = DataLoader(ds_train, **base_kw, collate_fn=_MixCut())
```
*   **`collate_fn`**: Đóng vai trò là một bộ lọc trung gian. Trước khi một batch ảnh đáy mắt võng mạc được đưa vào mạng, `_MixCut` sẽ rút thăm: $50\%$ cơ hội áp dụng MixUp (trộn độ sáng và nhãn) và $50\%$ cơ hội áp dụng CutMix (cắt dán phân vùng tổn thương).

### 2.3. Khởi Tạo Mô Mô Hình Đa Nhiệm và Bộ Tối Ưu Hóa (Optimizers)
```python
        # Khởi tạo backbone qua hàm build_model và đẩy lên GPU CUDA
        model = build_model(
            model_type=model_type, pretrained=True,
            img_size=img_size,
            variant=cfg.get('model', {}).get('variant', 'tiny')
        ).to(device)

        pw        = pos_weight.to(device)
        # Khởi tạo loss đa nhiệm y sinh MultiTaskLoss kết hợp pos_weight
        criterion = MultiTaskLoss(pos_weight=pw, lam=cfg['loss'].get('lam', 0.1))
        
        # Khởi tạo AdamW với trọng số suy giảm (weight decay) chống quá khớp
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['optimizer'].get('lr', 3e-4),
            weight_decay=cfg['optimizer'].get('weight_decay', 0.01)
        )
        
        # Điều tốc tốc độ học theo chu kỳ đường cong Cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
```

### 2.4. Vòng Lặp Epoch Chính (`run_epoch`)
Hàm con `run_epoch` quản lý quá trình lan truyền tiến và đạo hàm lùi cho một vòng lặp (1 Epoch) huấn luyện:

```python
        def run_epoch(loader, mode, threshold=0.5):
            model.train() if mode == 'train' else model.eval()
            tot = 0; probs = []; tgts = []; ap = []; at = []
            ctx = torch.enable_grad() if mode == 'train' else torch.no_grad()
            with ctx:
                for batch_data in loader:
                    imgs = batch_data['image'].to(device)
                    lbl  = batch_data['labels'].to(device)
                    age  = batch_data['age'].to(device)
                    
                    # 1. Forward Pass qua mô hình để lấy logits bệnh và age_pred
                    out  = model(imgs)
                    
                    # 2. Tính loss đa nhiệm kết hợp
                    loss, _ = criterion(out['logits'], lbl, out['age_pred'], age)
                    
                    # 3. Lan truyền đạo hàm lùi (Backward Pass) và cập nhật AdamW
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Tránh bùng nổ gradient
                        optimizer.step()
                        
                    tot += loss.item() * imgs.size(0)
                    probs.extend(torch.sigmoid(out['logits']).detach().cpu().tolist())
                    tgts.extend(lbl.detach().cpu().tolist())
                    ap.extend(out['age_pred'].squeeze(1).detach().cpu().tolist())
                    at.extend(age.squeeze(1).detach().cpu().tolist())
```
*   **Tính toán Metrics (F1-macro, AUC-ROC, Age MAE):** Cuối mỗi epoch, hàm tự động khôi phục tuổi thực tế từ Z-score chuẩn hóa để tính MAE (năm tuổi), đồng thời tính toán điểm F1-macro của 8 nhãn dựa trên ngưỡng quyết định (`threshold`) truyền vào.

### 2.5. Tìm Ngưỡng Động Tối Ưu (Dynamic Thresholding)
Sau khi kết thúc quá trình chạy epochs huấn luyện chính, mô hình tốt nhất (`best.pth`) sẽ được nạp lại để quét tìm ngưỡng quyết định tối ưu:

```python
        # 1. Tìm ngưỡng động tối ưu trên Validation
        val_preds = run_epoch(val_loader, 'val')
        from src.utils import find_best_thresholds, get_label_names
        
        # Gọi hàm find_best_thresholds để quét tìm bộ ngưỡng giúp tối đa hóa F1-macro trên tập Val
        best_thresholds = find_best_thresholds(
            torch.FloatTensor(val_preds['probs']),
            torch.FloatTensor(val_preds['targets'])
        )
```
*   **Ý nghĩa:** Giải pháp này giúp điều chỉnh linh hoạt độ nhạy chẩn đoán của mô hình. Các bệnh rất hiếm sẽ nhận được ngưỡng quyết định thấp hơn `0.5` để tăng khả năng phát hiện bệnh lý lâm sàng.

### 2.6. Đánh Giá Đối Chứng và Lưu Kết Quả Cuối Cùng
Mô hình được đánh giá song song trên tập kiểm thử độc lập (TEST SET) dưới 2 cấu hình ngưỡng để ghi nhận sự cải tiến:

```python
        # Đánh giá Test với ngưỡng mặc định cố định 0.5
        test_m_default = run_epoch(test_loader, 'test', threshold=0.5)

        # Đánh giá Test với bộ ngưỡng động tối ưu vừa tìm được
        test_m_opt = run_epoch(test_loader, 'test', threshold=best_thresholds)

        # Đóng gói và kết xuất toàn bộ log huấn luyện ra file kết quả kết quả JSON
        result = {
            'exp': exp_name,
            'best_val_f1_default': best_f1,
            'optimal_thresholds': best_thresholds,
            'test_default_0.5': test_m_default,
            'test_optimal_dynamic': test_m_opt,
            'log': log
        }
        json.dump(result, open(f'{out_dir}/results.json', 'w'), indent=2, default=str)
```

---

*Tài liệu hướng dẫn giải thích hàm run_experiment được biên soạn bởi Antigravity nhằm hỗ trợ Ngô Đình Đạt hiện thực hóa Đồ án Tốt nghiệp xuất sắc.*
