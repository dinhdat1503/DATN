# Hành trình hiểu mô hình — Từ ảnh đến chẩn đoán (CNN & Swin)

> Tài liệu này tổng hợp toàn bộ hành trình tìm hiểu mô hình theo trình tự dòng dữ liệu, mỗi bước gồm 4 phần:
> **💡 Khái niệm → 📄 Code → 📁 File → 🔍 Giải thích logic → 🔗 Mạch nối.**
>
> Bài toán: phân loại nhị phân ảnh đáy mắt ODIR-5K (**Normal vs Pathological**) bằng mạng **Siamese 2 mắt**, backbone **CNN (EfficientNet-B0)** hoặc **Swin-Tiny**, kèm nhánh phụ dự đoán tuổi.
>
> Tài liệu bạn đọc kèm: [`LY_THUYET_CNN_SWIN_NEN_TANG.md`](LY_THUYET_CNN_SWIN_NEN_TANG.md) · [`GIAI_THICH_MO_HINH_CNN_SWIN.md`](GIAI_THICH_MO_HINH_CNN_SWIN.md) · [`GIAI_THICH_CODE_HUAN_LUYEN_CNN_SWIN.md`](GIAI_THICH_CODE_HUAN_LUYEN_CNN_SWIN.md)

---

## Mục lục
- [0. Sơ đồ tổng thể + 2 giai đoạn](#0-sơ-đồ-tổng-thể--2-giai-đoạn)
- [1. Đầu vào: nạp 2 ảnh đã tiền xử lý](#1-đầu-vào-nạp-2-ảnh-đã-tiền-xử-lý)
- [2. Backbone: trích đặc trưng (CNN/Swin)](#2-backbone-trích-đặc-trưng-cnnswin)
- [3. Vector đặc trưng](#3-vector-đặc-trưng)
- [4. Siamese: 2 mắt qua chung 1 backbone](#4-siamese-2-mắt-qua-chung-1-backbone)
- [5. Ghép 2 vector (concat)](#5-ghép-2-vector-concat)
- [6. Fusion MLP: trộn 2 mắt](#6-fusion-mlp-trộn-2-mắt)
- [7. Head + sigmoid → dự đoán bệnh](#7-head--sigmoid--dự-đoán-bệnh)
- [8. Nhánh tuổi (nhiệm vụ phụ trợ)](#8-nhánh-tuổi-nhiệm-vụ-phụ-trợ)
- [9. Loss: đo sai bao nhiêu](#9-loss-đo-sai-bao-nhiêu)
- [10. w + backward: cách mạng học](#10-w--backward-cách-mạng-học)
- [11. Hai giai đoạn: huấn luyện vs dự đoán](#11-hai-giai-đoạn-huấn-luyện-vs-dự-đoán)
- [12. Đánh giá: Test + các thước đo](#12-đánh-giá-test--các-thước-đo)
- [13. So sánh 6 thí nghiệm → chọn mô hình](#13-so-sánh-6-thí-nghiệm--chọn-mô-hình)
- [Phụ lục: Câu hỏi bảo vệ](#phụ-lục-câu-hỏi-bảo-vệ)

---

## 0. Sơ đồ tổng thể + 2 giai đoạn

**Đường đi của 1 ảnh qua mô hình (lượt đi / forward):**
```
2 ảnh đã tiền xử lý
   → [dataset.py] nạp ảnh
   → [backbone CNN/Swin] trích đặc trưng → 2 vector
   → xóa mắt thiếu → [concat] ghép → [fusion MLP] trộn → vector 512
   → ┬─ [head bệnh] → sigmoid → xác suất → Normal/Pathological
     └─ [head tuổi] → age_pred (nhiệm vụ phụ)
```

**Mô hình có 2 giai đoạn lớn:**
```
╔═ GIAI ĐOẠN 1: HUẤN LUYỆN (ảnh CÓ đáp án) ═╗   lặp nghìn lần:
║   dự đoán → so đáp án → loss → sửa w        ║   (tìm bộ w tốt)
╚═════════════════════════════════════════════╝
                 │  train xong → best_model.pth
                 ▼
╔═ GIAI ĐOẠN 2: DỰ ĐOÁN (ảnh MỚI, KHÔNG đáp án) ═╗
║   chỉ forward → ra kết luận. Không loss/sửa w   ║
╚═════════════════════════════════════════════════╝
```

> Ví von xuyên suốt: **huấn luyện = luyện đề có đáp án** (làm bài → xem đáp án → rút kinh nghiệm); **dự đoán = thi đề mới** (chỉ làm bài, không ai chấm để sửa).

---

## 1. Đầu vào: nạp 2 ảnh đã tiền xử lý

### 💡 Khái niệm
Ảnh đã được tiền xử lý (crop + Ben-Graham + CLAHE) và **lưu sẵn trên đĩa** ở pha chuẩn bị dữ liệu. Khi train, file `dataset.py` mới **đọc lại** ảnh, **ghép cặp 2 mắt theo từng bệnh nhân**, xử lý mắt thiếu, và chuẩn hóa — biến thành "batch" để đưa vào mạng.

### 📄 Code & 📁 File
**File:** [`src/dataset.py`](../src/dataset.py) — class `BinocularDataset`
```python
# Ghép cặp 2 mắt theo Patient ID (mỗi bệnh nhân = 1 mẫu)
for patient_id, group in self.df.groupby("ID"):
    first = group.iloc[0]
    age = float(first["Patient Age"])
    label = 1 - int(first["N"])           # N=1 (Normal)→0 ; N=0 (bệnh)→1
    left_fn = right_fn = None
    for _, row in group.iterrows():
        fn = str(row["filename"])
        if "_left" in fn:  left_fn = fn
        elif "_right" in fn: right_fn = fn
    self.patients.append({"patient_id": ..., "age": age, "label": label,
                          "left_filename": left_fn, "right_filename": right_fn})

# Đọc 1 ảnh mắt; nếu thiếu mắt → ảnh đen + cờ missing
def _load_eye(self, filename):
    if filename is None:
        return torch.zeros(3, self.img_size, self.img_size), True   # mắt thiếu
    img = cv2.imread(str(self.img_dir / filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = self.transforms(image=img)["image"]                    # resize+normalize
    return tensor, False
```
**File:** [`src/transforms.py`](../src/transforms.py) — chuẩn hóa ảnh
```python
A.Resize(img_size, img_size),
A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),   # chuẩn theo ImageNet
ToTensorV2(),
```

### 🔍 Giải thích logic
- **`groupby("ID")`** — ODIR-5K lưu mỗi ảnh 1 dòng; gộp theo `ID` để **1 bệnh nhân = 1 mẫu** gồm cặp mắt trái/phải.
- **`label = 1 - int(first["N"])`** — cột `N` là Normal (1=khỏe). Đảo lại để **0=Normal, 1=Pathological** (đây chính là "đáp án thật" dùng tính loss sau này).
- **`_load_eye`** — đọc ảnh bằng OpenCV, đổi BGR→RGB, áp transform. **Mắt thiếu → trả ảnh đen + cờ `missing=True`** để mạng biết bỏ qua.
- **`Normalize(ImageNet)`** — đưa ảnh về cùng "thang đo" mà backbone pretrained ImageNet quen thuộc → học hiệu quả hơn.

### 🔗 Mạch nối
- **Trước:** ảnh enhanced nằm sẵn trên đĩa (pha chuẩn bị dữ liệu).
- **Sau:** mỗi batch (gồm `left_image`, `right_image`, `left_missing`, `right_missing`, `label`, `age`) được đưa vào model ở Bước 2.

---

## 2. Backbone: trích đặc trưng (CNN/Swin)

### 💡 Khái niệm
**Backbone = "xương sống"** — phần chính "nhìn" ảnh và **tóm tắt thành vector đặc trưng**. Là "đôi mắt đọc ảnh", lấy sẵn từ thư viện **timm**, có thể là **CNN** hoặc **Swin**. Đổi giữa 2 cái chỉ là đổi `model_type`.

### 📄 Code & 📁 File
**File:** [`src/models/backbone.py`](../src/models/backbone.py) — hàm `build_backbone()`
```python
def build_backbone(model_type="cnn", pretrained=True, img_size=384):
    import timm
    if model_type in ("cnn", "efficientnet", "efficientnet_b0"):
        backbone = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                     num_classes=0, global_pool="avg")
        feature_dim = backbone.num_features   # = 1280
        return backbone, feature_dim
    if model_type in ("swin", "swin_tiny", "swin_transformer"):
        kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
        if img_size != 224:
            kwargs["img_size"] = img_size     # Swin cần biết kích thước ảnh
        backbone = timm.create_model("swin_tiny_patch4_window7_224", **kwargs)
        feature_dim = backbone.num_features   # = 768
        return backbone, feature_dim
```

### 🔍 Giải thích logic
- **`import timm`** — mượn thư viện chứa sẵn mạng + trọng số pretrained (không tự code mạng).
- **`if model_type in (...)`** — "công tắc": `"cnn"` rẽ EfficientNet, `"swin"` rẽ Swin.
- **`num_classes=0`** — ⭐ **cắt bỏ đầu phân loại 1000 lớp ImageNet**, chỉ giữ phần trích đặc trưng.
- **`global_pool="avg"`** — gộp bản đồ đặc trưng thành **1 vector**.
- **`feature_dim = backbone.num_features`** — số chiều vector do **kiến trúc quyết định**: CNN=1280, Swin=768.
- **`if img_size != 224`** — Swin gốc học ở 224; chạy 384 phải báo để timm nội suy bảng vị trí. CNN không cần (tích chập bất biến kích thước).

### 🔗 Mạch nối
- **Hàm này chạy khi:** khởi tạo model (1 lần), do `siamese.py` gọi: `self.backbone = build_backbone(...)`.
- **Sau:** backbone tạo ra được dùng trong `forward` để biến ảnh → vector (Bước 3).

---

## 3. Vector đặc trưng

### 💡 Khái niệm
Đầu ra của backbone cho **mỗi ảnh** là **1 vector đặc trưng** — một dãy số cô đọng nội dung con mắt. Backbone **nén** ảnh (≈442.000 pixel) xuống còn **1280 số (CNN)** hoặc **768 số (Swin)**. Mỗi số là một đặc trưng trừu tượng mà mạng **tự học**.

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py) — trong `forward()`
```python
left_feat  = self.backbone(left_image)    # [B, D]  D=1280 (CNN) / 768 (Swin)
right_feat = self.backbone(right_image)   # [B, D]
```

### 🔍 Giải thích logic
- Gọi `self.backbone(ảnh)` → trả về tensor `[B, D]`: `B` = số ảnh trong batch, `D` = số chiều đặc trưng.
- 2 mắt → 2 vector riêng (`left_feat`, `right_feat`), cùng kích thước vì cùng 1 backbone.
- Bạn **không đọc trực tiếp** từng số được — nhưng gộp `D` số lại đủ để phân biệt mắt khỏe vs bệnh.

### 🔗 Mạch nối
- **Trước:** ảnh từ `dataset.py` (Bước 1) qua backbone (Bước 2).
- **Sau:** 2 vector này được xử lý mắt thiếu rồi ghép (Bước 5).

---

## 4. Siamese: 2 mắt qua chung 1 backbone

### 💡 Khái niệm
**Siamese (mạng "song sinh") = cho 2 đầu vào (2 mắt) đi qua CÙNG MỘT backbone dùng chung trọng số, rồi ghép kết quả để chẩn đoán.** Không phải 2 mạng riêng — mà **1 mạng dùng 2 lần**. Cả lớp `BinocularClassifier` chính là mạng Siamese; backbone chỉ là 1 phần bên trong.

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py) — class `BinocularClassifier`
```python
class BinocularClassifier(nn.Module):
    def __init__(self, backbone_type="cnn", ...):
        super().__init__()
        self.backbone, self.feature_dim = build_backbone(...)   # CHỈ 1 backbone

    def forward(self, left_image, right_image, left_missing, right_missing):
        left_feat  = self.backbone(left_image)    # dùng lần 1
        right_feat = self.backbone(right_image)   # dùng lần 2  ← chung trọng số
```

### 🔍 Giải thích logic
- **Chỉ tạo `self.backbone` MỘT lần** trong `__init__` → đây là bản chất "chia sẻ trọng số".
- Trong `forward`, gọi **cùng** `self.backbone` cho cả 2 mắt → 2 mắt được "đọc" theo **cùng một cách**.
- Lợi ích: ít tham số (đỡ overfit), nhất quán giữa 2 mắt, hợp thực tế (chẩn đoán cho cả bệnh nhân).

### 🔗 Mạch nối
- **Trước:** ảnh 2 mắt (Bước 1).
- **Sau:** 2 vector đặc trưng → xử lý mắt thiếu + ghép (Bước 5).

---

## 5. Ghép 2 vector (concat)

### 💡 Khái niệm
Trước khi ghép, mạng **xóa nhiễu mắt thiếu** (nhân vector ×0). Sau đó **"ghép" = nối đuôi 2 vector** thành 1 vector dài gấp đôi — vì chẩn đoán cho **cả bệnh nhân** nên phải nhìn 2 mắt cùng lúc.

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py) — trong `forward()`
```python
# Bước 2: xóa mắt thiếu (mắt thiếu → vector 0)
left_mask  = (~left_missing).float().unsqueeze(1)   # 1=có mắt, 0=thiếu
left_feat  = left_feat  * left_mask
right_feat = right_feat * right_mask

# Bước 3: ghép 2 vector
fused = torch.cat([left_feat, right_feat], dim=-1)  # [B, 2D]
```

### 🔍 Giải thích logic
- **`~left_missing`** — dấu `~` đảo bit: mắt thiếu (`True`) → `False` → `.float()`=0 → nhân làm vector về 0 (xóa nhiễu ảnh đen). Mắt có (`False`) → mask=1 → giữ nguyên.
- **`torch.cat([..], dim=-1)`** — nối 2 vector thành 1 vector dài `2D` (CNN: 1280+1280=2560). Đây chỉ là **xếp cạnh nhau**, chưa tính toán/học gì.

### 🔗 Mạch nối
- **Trước:** 2 vector đặc trưng (Bước 3).
- **Sau:** vector ghép `[B, 2D]` đi vào Fusion MLP để "trộn" (Bước 6).

---

## 6. Fusion MLP: trộn 2 mắt

### 💡 Khái niệm
**Fusion MLP = một mạng nơ-ron nhỏ** đặt sau bước ghép, **trộn và học cách kết hợp** đặc trưng 2 mắt (2D số) thành "góc nhìn chung" (512 số). **MLP** = mạng nơ-ron cơ bản gồm các lớp `Linear` (mỗi nơ-ron tính tổng có trọng số của tất cả đầu vào). Đây là chỗ thật sự "suy luận", khác với `concat` chỉ xếp cạnh nhau.

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py) — trong `__init__()`
```python
self.fusion_mlp = nn.Sequential(
    nn.Linear(2 * self.feature_dim, 512),   # 2560 → 512 (TRỘN 2 mắt, có học)
    nn.LayerNorm(512),                      # chuẩn hóa → train ổn định
    nn.SiLU(),                              # hàm kích hoạt phi tuyến
    nn.Dropout(p=dropout),                  # tắt ngẫu nhiên 30% → chống overfit
)
# trong forward:
fused = self.fusion_mlp(fused)              # [B, 2D] → [B, 512]
```

### 🔍 Giải thích logic
- **`nn.Linear(2560, 512)`** — 512 nơ-ron, **mỗi nơ-ron nhìn cả 2 mắt** (2560 số) rồi tổng hợp thành 1 số. Trọng số `w` của các nơ-ron này **tự học** cách cân nhắc trái/phải.
- **`LayerNorm`** — cân chỉnh con số cho cân đối → huấn luyện mượt.
- **`SiLU`** — thêm tính phi tuyến → học được quan hệ phức tạp.
- **`Dropout(0.3)`** — mỗi lần train bỏ ngẫu nhiên 30% nơ-ron → buộc mạng học chắc, chống học vẹt.

### 🔗 Mạch nối
- **Trước:** vector ghép `[B, 2D]` (Bước 5).
- **Sau:** vector 512 "góc nhìn chung" → 2 head (Bước 7 + 8).

---

## 7. Head + sigmoid → dự đoán bệnh

### 💡 Khái niệm
**Head bệnh** ép vector 512 thành **1 số (logit)** — số thô chưa đọc được. **Sigmoid** biến logit → **xác suất bệnh (0–1)**. So với ngưỡng (0.5 hoặc Youden) → kết luận **Normal/Pathological**. Đây chính là **dự đoán** của mạng — luôn được tạo **trước** khi tính loss.

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py) — head; **File:** [`src/engine.py`](../src/engine.py) — sigmoid
```python
# siamese.py — __init__ + forward
self.classification_head = nn.Linear(512, 1)
logits = self.classification_head(fused)      # 512 → 1 (số thô "logit")

# engine.py — biến logit thành xác suất
probs = torch.sigmoid(logits)                 # logit → xác suất 0..1
```

### 🔍 Giải thích logic
- **`Linear(512, 1)`** — gom 512 số "góc nhìn chung" thành **1 con số** = điểm số bệnh (logit).
- **`sigmoid`** — ép mọi số về (0,1): `logit=2.5 → 0.92` (92% bệnh); `logit=-1.8 → 0.14`.
- So ngưỡng: `> ngưỡng → Pathological`, `< ngưỡng → Normal`.

### 🔗 Mạch nối
- **Trước:** vector 512 (Bước 6).
- **Sau:** xác suất này được đem **so với đáp án thật** để tính loss (Bước 9).

---

## 8. Nhánh tuổi (nhiệm vụ phụ trợ)

### 💡 Khái niệm
Mạng có **head thứ 2 chạy song song**: **dự đoán tuổi**. Đây là **bài toán hồi quy** (đoán 1 con số, không qua sigmoid), và là **nhiệm vụ phụ trợ (multi-task)** — không phải mục tiêu chính, mà để **buộc backbone học đặc trưng võng mạc giàu hơn**, gián tiếp giúp nhánh bệnh chính xác hơn.

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py)
```python
self.regression_head = nn.Linear(512, 1)        # đầu ra tuổi (không sigmoid)
# forward trả về cả 2:
return {
    "logits":   self.classification_head(fused),  # bệnh
    "age_pred": self.regression_head(fused),       # tuổi
}
```

### 🔍 Giải thích logic
- Cả 2 head cùng nhìn vector 512, nhưng head tuổi trả **thẳng 1 con số** (tuổi đã chuẩn hóa) — không qua sigmoid vì đây là hồi quy, không phải phân loại.
- Vì sao giúp ích: tuổi ảnh hưởng cấu trúc võng mạc; bắt mạng đoán cả tuổi → hiểu võng mạc sâu hơn → đoán bệnh tốt hơn (giống học giỏi Toán giúp tư duy Lý tốt hơn).

### 🔗 Mạch nối
- **Trước:** vector 512 (Bước 6).
- **Sau:** `age_pred` góp vào loss với hệ số nhỏ 0.05 (Bước 9).

---

## 9. Loss: đo sai bao nhiêu

### 💡 Khái niệm
**Loss = con số đo mạng đang sai bao nhiêu** so với đáp án thật. Cao = sai nhiều, thấp = gần đúng. Huấn luyện = **kéo loss xuống thấp nhất**. Trong dự án, loss = **Loss bệnh (Focal)** + 0.05 × **Loss tuổi (SmoothL1)**.

### 📄 Code & 📁 File
**File:** [`src/losses.py`](../src/losses.py) — `MultiTaskLoss`
```python
cls_loss = self.cls_loss_fn(logits, labels)      # Focal Loss (bệnh)
reg_loss = self.reg_loss_fn(age_pred, age_true)  # SmoothL1 (tuổi)
total = cls_loss + self.lam_age * reg_loss       # lam_age = 0.05
```

### 🔍 Giải thích logic
- **`cls_loss` (Focal Loss)** — đo sai cho phân loại, thiết kế để **xử lý mất cân bằng** (dữ liệu nhiều bệnh hơn khỏe) và tập trung vào mẫu khó.
- **`reg_loss` (SmoothL1)** — đo tuổi đoán lệch bao nhiêu năm, bền với outlier.
- **`total = cls + 0.05*reg`** — phần bệnh là chính; tuổi nhân 0.05 nên chỉ "góp ý nhẹ", không lấn át.

### 🔗 Mạch nối
- **Trước:** dự đoán (Bước 7+8) + đáp án thật (`label`, `age` từ Bước 1).
- **Sau:** loss này được dùng để tính gradient và sửa `w` (Bước 10).

---

## 10. w + backward: cách mạng học

### 💡 Khái niệm
**`w` (trọng số) = độ quan trọng của mỗi kết nối**. Ban đầu **ngẫu nhiên**, rồi **được tính toán điều chỉnh dần** qua huấn luyện. Mạng không biết từng `w` đúng/sai trực tiếp — nó đo **loss** rồi dùng **gradient (đạo hàm)** để biết *"chỉnh `w` này thì loss tăng hay giảm"*, rồi chỉnh ngược hướng đó. Lặp nghìn lần → `w` về giá trị tốt. **"Huấn luyện" = tìm bộ `w` tốt.**

### 📄 Code & 📁 File
**File:** [`src/models/siamese.py`](../src/models/siamese.py) — khởi tạo ngẫu nhiên; **File:** [`src/engine.py`](../src/engine.py) — học
```python
# siamese.py: w khởi tạo NGẪU NHIÊN (có kiểm soát)
nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

# engine.py: mỗi batch — TÍNH TOÁN cập nhật w
loss.backward()        # ④ tính mỗi w nên tăng/giảm bao nhiêu (gradient)
optimizer.step()       # ⑤ cập nhật tất cả w theo gradient
optimizer.zero_grad()  #   xóa gradient cũ, sẵn sàng vòng sau
```

### 🔍 Giải thích logic
- **`kaiming_uniform_`** — đặt `w` xuất phát ngẫu nhiên trong khoảng hợp lý (giúp học nhanh hơn).
- **`loss.backward()`** — "lần ngược" từ loss về từng `w`, tính **gradient** = mỗi `w` góp bao nhiêu vào sai số và nên đổi hướng nào (PyTorch tự làm cho cả triệu `w`).
- **`optimizer.step()`** — cập nhật: `w_mới = w_cũ − tốc_độ_học × gradient` (dấu trừ tự đẩy về phía loss nhỏ).
- Ví von: đi xuống thung lũng trong sương mù — cảm nhận độ dốc dưới chân (gradient) rồi bước xuống, lặp tới đáy.

### 🔗 Mạch nối
- **Trước:** loss (Bước 9).
- **Sau:** lặp lại với batch tiếp theo; sau nhiều epoch → `w` hội tụ → kết thúc giai đoạn huấn luyện (Bước 11).

---

## 11. Hai giai đoạn: huấn luyện vs dự đoán

### 💡 Khái niệm
- **Huấn luyện:** dùng ảnh **CÓ đáp án**. Mỗi vòng: **dự đoán → so đáp án → loss → sửa w**, lặp nghìn lần. Có **two-stage** (đóng băng backbone vài epoch đầu rồi mở khóa) và **early stopping** (dừng khi không cải thiện).
- **Dự đoán:** dùng ảnh **MỚI, KHÔNG đáp án**. Chỉ chạy forward → ra kết quả, **không loss, không sửa w**.

### 📄 Code & 📁 File
**File:** [`src/engine.py`](../src/engine.py) — `fit()`
```python
if two_stage:
    model.freeze_backbone()                 # Stage 1: khóa backbone, train head
...
if epoch == freeze_epochs + 1:
    model.unfreeze_backbone()               # Stage 2: mở khóa, fine-tune toàn mạng
...
if val_auc > best_auc:                      # cải thiện → lưu best_model.pth
    torch.save({...}, best_path)
else:
    no_improve += 1
    if no_improve >= patience: break        # EARLY STOPPING
```

### 🔍 Giải thích logic
- **`freeze_backbone` → `unfreeze_backbone`** — 5 epoch đầu chỉ train head (không phá pretrained), sau đó mới fine-tune cả mạng với LR nhỏ.
- **Early stopping theo `val_auc`** — theo dõi điểm trên tập val; nếu nhiều epoch liền không tốt hơn thì dừng → tránh overfit, lưu lại bản tốt nhất.

### 🔗 Mạch nối
- **Trước:** vòng học của Bước 10.
- **Sau:** model tốt nhất (`best_model.pth`) → đem đánh giá trên Test (Bước 12).

---

## 12. Đánh giá: Test + các thước đo

### 💡 Khái niệm
Sau khi train, đánh giá mô hình trên **tập Test** (ảnh **chưa từng thấy**) để biết nó **thật sự giỏi cỡ nào**. Dùng các thước đo y sinh: **AUC, Accuracy, Sensitivity (độ nhạy), Specificity (độ đặc hiệu), F1**. Chọn **ngưỡng Youden** tối ưu trên val rồi áp lên test. Kết quả lưu ra `test_results.json`.

### 📄 Code & 📁 File
**File:** [`src/engine.py`](../src/engine.py) — `evaluate_test()`; **File:** [`src/metrics.py`](../src/metrics.py) — chỉ số
```python
# engine.py: tìm ngưỡng tốt nhất trên val rồi đánh giá test
best_thresh = find_best_threshold(val_probs, val_targets)   # Youden
opt = compute_binary_metrics(test_probs, test_targets, threshold=best_thresh)
json.dump(results, ...)   # → results/<exp>/test_results.json

# metrics.py: Youden J = Sensitivity + Specificity − 1
j = sens + spec - 1.0
```

### 🔍 Giải thích logic
- **Vì sao Test riêng?** ảnh mới hoàn toàn → đo "giỏi thật" chứ không phải học vẹt đề luyện.
- **`find_best_threshold` (Youden)** — quét ngưỡng tìm điểm cân bằng độ nhạy/đặc hiệu tốt nhất (quan trọng y tế: không bỏ sót bệnh).
- **`compute_binary_metrics`** — tính AUC/F1/Sensitivity/Specificity từ confusion matrix.
- **Sensitivity** thường quan trọng nhất trong y tế: thà báo nhầm còn hơn bỏ sót người bệnh.

### 🔗 Mạch nối
- **Trước:** `best_model.pth` (Bước 11).
- **Sau:** mỗi thí nghiệm sinh 1 `test_results.json` → đem so sánh (Bước 13).

---

## 13. So sánh 6 thí nghiệm → chọn mô hình

### 💡 Khái niệm
Dự án có **6 thí nghiệm** (CNN/Swin × raw/enhanced/enhanced+aug). Script `evaluate.py` **gom 6 bảng điểm** (`test_results.json`) thành **1 bảng so sánh** để kết luận: kiến trúc nào tốt hơn, tiền xử lý/augmentation có giúp không. Đây là **kết quả chính** để viết báo cáo.

### 📄 Code & 📁 File
**File:** [`evaluate.py`](../evaluate.py)
```python
for exp in DEFAULT_ORDER:                       # 6 thí nghiệm
    res = load_result(exp / "test_results.json")
    m = res["metrics_threshold_optimal"]        # lấy điểm ở ngưỡng Youden
    # → ghi 1 dòng: Accuracy, AUC, F1, Sensitivity, Specificity, Age MAE
write(results_dir / "comparison_table.md")      # bảng so sánh
```

### 🔍 Giải thích logic
- **`evaluate.py` KHÔNG chạy model** — chỉ **đọc lại** các file JSON đã có rồi dựng bảng Markdown. (Khác `evaluate_test()` trong engine.py — cái đó mới thật sự chạy model tính metric.)
- Mỗi dòng = 1 thí nghiệm với đầy đủ chỉ số → dễ so sánh trực quan.

### 🔗 Mạch nối
- **Trước:** 6 file `test_results.json` (Bước 12).
- **Sau:** chọn mô hình điểm cao nhất → dùng chẩn đoán bệnh nhân thật (Giai đoạn 2, Bước 0).

---

## Phụ lục: Câu hỏi bảo vệ

| Câu hỏi | Trả lời nhanh |
|---|---|
| Đầu vào mô hình là gì? | Cặp ảnh mắt trái+phải của 1 bệnh nhân, đã tiền xử lý + chuẩn hóa ImageNet |
| Backbone là gì? | Phần trích đặc trưng (CNN/Swin từ timm), biến ảnh → vector |
| Có mấy mô hình? | **1** backbone, dùng **2 lần** cho 2 mắt (Siamese), KHÔNG phải 2 mô hình |
| Vector đặc trưng dài bao nhiêu? | CNN 1280, Swin 768 — do kiến trúc quyết định |
| "Ghép" 2 mắt là gì? | `concat` nối 2 vector, rồi Fusion MLP trộn lại |
| Mạng học bằng cách nào? | dự đoán → loss → backward (gradient) → chỉnh `w`, lặp nghìn lần |
| `w` ngẫu nhiên hay tính toán? | Khởi đầu ngẫu nhiên, sau đó được tính toán điều chỉnh qua huấn luyện |
| Loss là gì? | Con số đo độ sai; train = kéo loss xuống thấp nhất |
| Vì sao có nhánh tuổi? | Nhiệm vụ phụ trợ (multi-task) giúp backbone học đặc trưng tốt hơn |
| Vì sao 2 giai đoạn? | Huấn luyện (có đáp án, sửa w) ≠ dự đoán thật (ảnh mới, chỉ forward) |
| Đánh giá bằng gì? | AUC, Sensitivity, Specificity... trên tập Test chưa từng thấy |
| Đổi CNN↔Swin là đổi gì? | Chỉ đổi `model_type` trong config → thay backbone, phần còn lại y nguyên |

> **Thông điệp cốt lõi:** Backbone (CNN/Swin từ timm) là *đôi mắt trích đặc trưng*; mạng **Siamese** ghép 2 mắt → chẩn đoán; toàn bộ phần thiết kế Siamese, loss đa nhiệm, quy trình huấn luyện và so sánh là **đóng góp của đồ án**.
