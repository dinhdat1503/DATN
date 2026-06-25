# Giải thích mô hình học sâu: CNN & Swin Transformer

> Tài liệu giải thích chi tiết phần **mô hình học sâu** của đồ án ODIR-5K (phân loại nhị phân Normal vs Pathological trên ảnh đáy mắt).
> Toàn bộ code mô hình nằm trong thư mục [`src/models/`](../src/models/).

---

## Mục lục
1. [Tổng quan: 3 file, 3 vai trò](#1-tổng-quan-3-file-3-vai-trò)
   - 📖 [ĐỌC CODE TỪNG DÒNG (line-by-line) — dành cho người mới](#-đọc-code-từng-dòng-line-by-line--dành-cho-người-mới)
2. [`backbone.py` — Bộ trích đặc trưng CNN & Swin](#2-backbonepy--bộ-trích-đặc-trưng-cnn--swin)
3. [So sánh cách CNN và Swin "nhìn" ảnh](#3-so-sánh-cách-cnn-và-swin-nhìn-ảnh)
4. [`siamese.py` — Mạng Siamese ghép 2 mắt](#4-siamesepy--mạng-siamese-ghép-2-mắt)
5. [`__init__.py` — Cổng vào `build_model`](#5-__init__py--cổng-vào-build_model)
6. [Sơ đồ kiến trúc tổng thể](#6-sơ-đồ-kiến-trúc-tổng-thể)
7. [Luồng hoạt động chi tiết (forward pass)](#7-luồng-hoạt-động-chi-tiết-forward-pass)
8. [Luồng huấn luyện 2 giai đoạn (two-stage)](#8-luồng-huấn-luyện-2-giai-đoạn-two-stage)
9. [Bảng tham số nhanh](#9-bảng-tham-số-nhanh)
10. [Câu hỏi thường gặp khi bảo vệ](#10-câu-hỏi-thường-gặp-khi-bảo-vệ)

---

## 1. Tổng quan: 3 file, 3 vai trò

| File | Vai trò |
|------|---------|
| `src/models/backbone.py` | **Nhà máy backbone** — tạo bộ trích đặc trưng: CNN (EfficientNet-B0) hoặc Swin Transformer-Tiny |
| `src/models/siamese.py`  | **Mạng Siamese** — ghép 2 mắt qua backbone chia sẻ trọng số → phân loại nhị phân |
| `src/models/__init__.py` | **Cổng vào** — hàm `build_model()` để các file khác gọi |

> **Điểm mấu chốt:** CNN và Swin **KHÔNG phải hai mạng khác nhau**. Chúng chỉ khác nhau ở **backbone** (phần xương sống trích đặc trưng). Phần còn lại của mạng (ghép 2 mắt, các đầu ra) **dùng chung y hệt**. Chỉ cần đổi một chuỗi `"cnn"` ↔ `"swin"` là chuyển mô hình.

---

## 📖 ĐỌC CODE TỪNG DÒNG (line-by-line) — dành cho người mới

> Phần này dành cho bạn **đã hiểu lý thuyết nhưng chưa map được vào code**. Mỗi đoạn code kèm giải thích tiếng Việt và nối thẳng với lý thuyết (backbone = đôi mắt, vector đặc trưng, Siamese = backbone chia sẻ...). Các cú pháp PyTorch được giải nghĩa ngay khi xuất hiện.

### Quy tắc vàng khi đọc model PyTorch
Một mạng trong PyTorch luôn là một **class** kế thừa `nn.Module`, gồm **2 phần**:
- `__init__` = **"phòng kho"**: khai báo có những bộ phận gì (backbone, MLP, các head). Chạy **1 lần** lúc tạo model.
- `forward` = **"dây chuyền"**: dữ liệu chạy qua các bộ phận theo thứ tự nào. Chạy **mỗi lần** đưa ảnh vào.

> 💡 Hình dung: `__init__` mua sẵn linh kiện; `forward` lắp chúng lại khi có hàng tới. **Luôn đọc `__init__` trước, rồi mới đọc `forward`.**

---

### BÀI 1 — `backbone.py`: "nhà máy" tạo đôi mắt

File này chỉ có **1 việc**: đẻ ra cái backbone (CNN hoặc Swin) biến 1 ảnh mắt → 1 vector.

**Khúc 1 — Định nghĩa hàm**
```python
def build_backbone(
    model_type: str = "cnn",      # chọn "cnn" hay "swin"
    pretrained: bool = True,      # có nạp trọng số học sẵn từ ImageNet không
    img_size: int = 384,          # ảnh vào to bao nhiêu (384×384)
) -> tuple[nn.Module, int]:       # trả về: (mạng, số chiều vector đặc trưng)
```
- `def build_backbone(...)` = khai báo một **hàm**.
- `model_type: str = "cnn"` → tham số kiểu chữ (`str`), **mặc định** `"cnn"`. Không truyền gì thì tự là `"cnn"`.
- `-> tuple[nn.Module, int]` → hàm **trả về** một cặp: `nn.Module` (một mạng) và `int` (số chiều vector = `feature_dim`).
- 💡 `: str`, `: int`, `-> ...` chỉ là **chú thích kiểu** cho dễ đọc, Python không bắt buộc.

**Khúc 2 — Mượn thư viện timm**
```python
import timm
```
Ta **không tự code** EfficientNet/Swin (quá phức tạp) mà mượn **timm** — thư viện chứa sẵn các mạng này kèm trọng số pretrained.

**Khúc 3 — Nhánh CNN**
```python
if model_type in ("cnn", "efficientnet", "efficientnet_b0"):
    backbone = timm.create_model(
        "efficientnet_b0",   # tên mạng muốn tạo
        pretrained=pretrained,
        num_classes=0,       # ⟵ BỎ tầng phân loại 1000 lớp của ImageNet
        global_pool="avg",   # ⟵ gộp bản đồ đặc trưng thành 1 vector
    )
    feature_dim = backbone.num_features   # = 1280
    return backbone, feature_dim
```
- `num_classes=0` = lý thuyết **"cắt bỏ đầu phân loại gốc"** — ta chỉ cần vector đặc trưng, không cần 1000 lớp ImageNet.
- `global_pool="avg"` = phép **pooling** — gộp bản đồ đặc trưng thành **một vector duy nhất**.
- `feature_dim = backbone.num_features` → hỏi mạng "vector dài bao nhiêu?" → CNN trả lời **1280**.
- `return backbone, feature_dim` → giao ra cặp (mạng, 1280).

**Khúc 4 — Nhánh Swin (gần giống, chỉ khác chỗ tô đậm)**
```python
if model_type in ("swin", "swin_tiny", "swin_transformer"):
    kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
    if img_size != 224:               # ⟵ ĐIỂM ĐẶC BIỆT CỦA SWIN
        kwargs["img_size"] = img_size
    backbone = timm.create_model("swin_tiny_patch4_window7_224", **kwargs)
    feature_dim = backbone.num_features   # = 768
    return backbone, feature_dim
```
- `if img_size != 224:` → Swin gốc học ở 224; chạy 384 nên phải báo `img_size` để timm **nội suy lại bảng vị trí (Relative Position Bias)**. CNN không cần vì tích chập bất biến kích thước.
- Swin trả về vector **768** chiều (nhỏ hơn CNN nhưng vai trò y hệt).

**📊 Sơ đồ file `backbone.py`**
```
build_backbone(model_type)
        │
        ├── "cnn"  → timm tạo EfficientNet-B0 → trả (mạng, 1280)
        │
        └── "swin" → timm tạo Swin-Tiny       → trả (mạng, 768)
                       (nếu img_size≠224 thì báo img_size)
```
→ Đưa vào 1 chữ `"cnn"`/`"swin"` → nhận ra 1 "đôi mắt" + biết vector dài bao nhiêu. **Hết.**

---

### BÀI 2 — `siamese.py`: lắp đôi mắt thành mạng hoàn chỉnh

#### 2A. Phần `__init__` — khai báo bộ phận

```python
class BinocularClassifier(nn.Module):       # mạng của ta, kế thừa nn.Module
    def __init__(self, backbone_type="cnn", pretrained=True, img_size=384, dropout=0.3):
        super().__init__()                  # bắt buộc: khởi động "động cơ" nn.Module
```
- `class ... (nn.Module)`: mọi mạng PyTorch đều kế thừa `nn.Module`.
- `super().__init__()`: dòng thủ tục bắt buộc, cứ luôn có ở đầu.

```python
        self.backbone, self.feature_dim = build_backbone(   # ⟵ gọi BÀI 1
            model_type=backbone_type, pretrained=pretrained, img_size=img_size
        )
```
- Gọi đúng hàm ở **Bài 1** để lấy "đôi mắt" + số chiều.
- `self.` nghĩa là **"của riêng model này"** (giống túi đồ của object) → lưu lại để `forward` dùng.
- 🔑 **CHỖ NÀY LÀ "SIAMESE":** chỉ tạo **MỘT** `self.backbone`. Lát nữa cả mắt trái và phải đều xài chung nó → **chia sẻ trọng số**.

```python
        self.fusion_mlp = nn.Sequential(            # "bộ não nhỏ" gộp 2 mắt
            nn.Linear(2 * self.feature_dim, 512),   # ghép 2 vector (2×D) → nén còn 512
            nn.LayerNorm(512),                      # chuẩn hóa cho ổn định
            nn.SiLU(),                              # hàm kích hoạt phi tuyến
            nn.Dropout(p=dropout),                  # tắt ngẫu nhiên 30% nơ-ron → chống học vẹt
        )
```
- `nn.Sequential(...)` = **xếp các lớp nối tiếp**, dữ liệu chảy qua lần lượt từ trên xuống.
- `nn.Linear(2*D, 512)` = lớp **fully-connected**: nhận vector `2*D` (CNN: 2×1280=2560) → biến thành 512. Đây là phần **Fusion** (gộp 2 mắt).

```python
        self.classification_head = nn.Linear(512, 1)   # đầu ra 1: xác suất BỆNH
        self.regression_head    = nn.Linear(512, 1)   # đầu ra 2: dự đoán TUỔI (phụ)
```
- Hai "cái đầu" cùng nhận vector 512 → mỗi cái cho ra **1 con số** (logit bệnh / tuổi).

**📊 Sơ đồ "phòng kho" (sau khi `__init__` chạy xong)**
```
self.backbone            ← 1 đôi mắt (CNN hoặc Swin), DÙNG CHUNG
self.fusion_mlp          ← bộ não gộp: (2D → 512)
self.classification_head ← Linear(512 → 1)  : bệnh
self.regression_head     ← Linear(512 → 1)  : tuổi
```
Lúc này **chưa có dữ liệu nào chạy** — mới chỉ "mua linh kiện".

#### 2B. Phần `forward` — dây chuyền 4 bước (TRÁI TIM)

```python
    def forward(self, left_image, right_image, left_missing, right_missing):
```

**BƯỚC 1 — Cho 2 mắt qua cùng backbone:**
```python
        left_feat  = self.backbone(left_image)    # [B, D]  vector đặc trưng mắt trái
        right_feat = self.backbone(right_image)   # [B, D]  vector đặc trưng mắt phải
```
- Gọi `self.backbone(...)` **2 lần** trên cùng một backbone → đúng tinh thần Siamese.
- `[B, D]`: `B` = số ảnh trong batch, `D` = 1280 (CNN) / 768 (Swin).

**BƯỚC 2 — Xóa nhiễu mắt bị thiếu:**
```python
        left_mask  = (~left_missing).float().unsqueeze(1)   # [B,1]  1=có mắt, 0=thiếu
        right_mask = (~right_missing).float().unsqueeze(1)
        left_feat  = left_feat  * left_mask    # mắt thiếu → vector thành toàn số 0
        right_feat = right_feat * right_mask
```
- Mắt thiếu = ảnh đen → đặc trưng "rác". Nhân với **0** để xóa.
- `~` = **đảo ngược**: `left_missing=True` (thiếu) → `~` thành `False` → `.float()` = `0.0` → nhân làm vector về 0. Mắt **không** thiếu → mask = 1.0 → giữ nguyên.

**BƯỚC 3 — Ghép 2 mắt rồi qua bộ não:**
```python
        fused = torch.cat([left_feat, right_feat], dim=-1)  # [B, 2D]  nối 2 vector
        fused = self.fusion_mlp(fused)                      # [B, 512] nén lại
```
- `torch.cat([...], dim=-1)` = **nối** 2 vector thành 1 vector dài gấp đôi (`2D`). CNN: 1280+1280 = 2560.

**BƯỚC 4 — Hai đầu ra:**
```python
        return {
            "logits":   self.classification_head(fused),   # [B,1] điểm số bệnh
            "age_pred": self.regression_head(fused),        # [B,1] tuổi
        }
```
- Trả về một **dict** chứa 2 kết quả. `logits` chưa phải xác suất — qua `sigmoid` mới thành xác suất 0–1.

**📊 Sơ đồ luồng `forward` (theo dõi shape — dùng CNN, D=1280)**
```
left_image  [B,3,384,384] ─┐
                           ├─► self.backbone ─► left_feat  [B,1280] ─┐
right_image [B,3,384,384] ─┘                    right_feat [B,1280] ─┤
                                                                     │ × mask (xóa mắt thiếu)
                                                                     ▼
                                          torch.cat ──► fused [B,2560]
                                                                     │
                                          fusion_mlp ──► fused [B,512]
                                                                     │
                                        ┌────────────────────────────┤
                                        ▼                            ▼
                          classification_head            regression_head
                              logits [B,1]                  age_pred [B,1]
                              (→ sigmoid → xác suất bệnh)      (tuổi)
```

#### 2C. Hai hàm phụ — phục vụ huấn luyện 2 giai đoạn
```python
    def freeze_backbone(self):                 # Stage 1: KHÓA đôi mắt
        for p in self.backbone.parameters():
            p.requires_grad = False            # không cho cập nhật trọng số backbone

    def unfreeze_backbone(self):               # Stage 2: MỞ khóa
        for p in self.backbone.parameters():
            p.requires_grad = True
```
- `parameters()` = tất cả trọng số học được của backbone.
- `requires_grad = False` = "đừng học/đừng cập nhật phần này" = **đóng băng backbone** trong two-stage training.

---

### 🎯 Bảng nối Lý thuyết ↔ Dòng code

| Lý thuyết bạn đã học | Nằm ở dòng code nào |
|---|---|
| Backbone = đôi mắt trích đặc trưng | `self.backbone = build_backbone(...)` |
| CNN → vector 1280, Swin → 768 | `feature_dim = backbone.num_features` |
| Siamese = chia sẻ trọng số | chỉ **1** `self.backbone`, gọi **2 lần** trong `forward` |
| Xóa mắt thiếu | `left_feat * left_mask` (Bước 2) |
| Gộp 2 mắt (Fusion) | `torch.cat(...)` → `fusion_mlp` (Bước 3) |
| Phân loại + tuổi | `classification_head` / `regression_head` (Bước 4) |
| Two-stage (đóng băng) | `freeze_backbone` / `unfreeze_backbone` |

---

## 2. `backbone.py` — Bộ trích đặc trưng CNN & Swin

Hàm duy nhất: `build_backbone(model_type, pretrained, img_size)` — trả về cặp `(module, feature_dim)`.

Dự án **không tự code mạng từ đầu** mà dùng thư viện **`timm`** (PyTorch Image Models) — thư viện chuẩn công nghiệp chứa sẵn hàng trăm kiến trúc kèm trọng số pretrained ImageNet.

### Nhánh CNN (`efficientnet_b0`)
```python
backbone = timm.create_model(
    "efficientnet_b0",
    pretrained=pretrained,
    num_classes=0,       # bỏ tầng phân loại gốc (1000 lớp ImageNet)
    global_pool="avg",   # global average pooling → 1 vector đặc trưng
)
feature_dim = backbone.num_features  # = 1280
```

### Nhánh Swin (`swin_tiny_patch4_window7_224`)
```python
kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
if img_size != 224:
    kwargs["img_size"] = img_size   # quan trọng với Swin!
backbone = timm.create_model("swin_tiny_patch4_window7_224", **kwargs)
feature_dim = backbone.num_features  # = 768
```

### Ba tham số `create_model` cần nắm
- `num_classes=0`: **cắt bỏ tầng phân loại cuối** của mạng gốc. Ta không cần 1000 lớp ImageNet, ta chỉ cần **vector đặc trưng**.
- `global_pool="avg"`: gộp bản đồ đặc trưng không gian thành **một vector** `[B, D]` bằng trung bình toàn cục.
- `pretrained=True`: nạp trọng số học từ ImageNet (transfer learning) — quan trọng vì dữ liệu y tế ít, học lại từ đầu sẽ kém.

### ⚠️ Điểm tinh tế: vì sao Swin cần `img_size` mà CNN không?
Swin-Tiny gốc huấn luyện ở **224×224**, nhưng đồ án chạy ở **384×384**. Khi đổi độ phân giải, **Relative Position Bias** (bảng độ lệch vị trí tương đối trong attention) không còn khớp kích thước → phải truyền `img_size=384` để `timm` **tự nội suy** bảng này.

CNN không gặp vấn đề này vì **tích chập bất biến với kích thước đầu vào** (cùng một bộ lọc trượt trên ảnh bất kỳ kích thước nào).

---

## 3. So sánh cách CNN và Swin "nhìn" ảnh

Đây là phần "luồng hoạt động bên trong" của hai backbone (chi tiết này nằm bên trong `timm`, không hiện trong code đồ án, nhưng cần hiểu để bảo vệ).

### EfficientNet-B0 (CNN) — tích chập, thu nhỏ dần
```
Ảnh [3, 384, 384]
   │  Stem Conv 3x3, stride 2
   ▼
[32, 192, 192]
   │  Chuỗi khối MBConv (Mobile Inverted Bottleneck)
   │  + Squeeze-Excitation, thu nhỏ không gian / tăng kênh dần
   ▼
[112, 24, 24] ──► [320, 12, 12]
   │  Conv 1x1 cuối → 1280 kênh
   ▼
[1280, 12, 12]
   │  Global Average Pooling (trung bình toàn bản đồ)
   ▼
Vector đặc trưng [1280]
```
**Bản chất:** bộ lọc cục bộ quét toàn ảnh, càng lên sâu "nhìn" được vùng càng rộng (receptive field lớn dần). Mạnh ở **đặc trưng cục bộ** (cạnh, kết cấu, tổn thương nhỏ).

### Swin-Tiny (Transformer) — patch + cửa sổ attention
```
Ảnh [3, 384, 384]
   │  Patch Embedding: chia ô 4x4, nhúng mỗi ô thành vector
   ▼
Token [96 chiều] × lưới 96x96
   │  Stage 1: 2 khối Swin Attention (cửa sổ 7x7 + shifted window)
   ▼
[96] × 96x96
   │  Patch Merging (gộp 2x2 ô → giảm lưới, tăng chiều)
   ▼  Stage 2: [192] × 48x48
   ▼  Stage 3: [384] × 24x24   (6 khối)
   ▼  Stage 4: [768] × 12x12
   │  Global Average Pooling
   ▼
Vector đặc trưng [768]
```
**Bản chất:** chia ảnh thành patch, tính **self-attention trong từng cửa sổ 7×7**, rồi **dịch cửa sổ (shifted window)** ở khối kế tiếp để thông tin "rò" sang cửa sổ lân cận. Mạnh ở **quan hệ toàn cục** và phụ thuộc xa giữa các vùng.

### Bảng đối chiếu
| | EfficientNet-B0 (CNN) | Swin-Tiny (Transformer) |
|---|---|---|
| Họ kiến trúc | Tích chập (convolution) | Self-attention cửa sổ trượt |
| Cách nhìn ảnh | Bộ lọc cục bộ, mở rộng dần | Patch → attention trong cửa sổ → dịch cửa sổ |
| Thế mạnh | Đặc trưng cục bộ, nhẹ, nhanh | Quan hệ toàn cục, phụ thuộc xa |
| `feature_dim` | **1280** | **768** |
| Độ phân giải input | Linh hoạt | Cần khai báo `img_size` |

---

## 4. `siamese.py` — Mạng Siamese ghép 2 mắt

Lớp `BinocularClassifier` hiện thực ý tưởng **Siamese (mạng Xiêm)**: mỗi bệnh nhân ODIR-5K có **2 ảnh đáy mắt (trái + phải)**, đưa cả hai qua **cùng một backbone (chia sẻ trọng số)** rồi gộp lại để chẩn đoán.

### 4.1 Khởi tạo `__init__`
```python
self.backbone, self.feature_dim = build_backbone(...)   # 1 backbone DUY NHẤT

self.fusion_mlp = nn.Sequential(
    nn.Linear(2 * self.feature_dim, 512),  # ghép 2 mắt: 2*D → 512
    nn.LayerNorm(512),
    nn.SiLU(),
    nn.Dropout(p=dropout),
)
self.classification_head = nn.Linear(512, 1)  # nhánh phân loại nhị phân
self.regression_head    = nn.Linear(512, 1)   # nhánh phụ: dự đoán tuổi
```

- **`self.backbone` chỉ có MỘT** → chính là "chia sẻ trọng số" (weight sharing) của Siamese. Mắt trái và phải đi qua **cùng bộ tham số**, nên model học một "cách nhìn mắt" chung, tiết kiệm tham số và tổng quát tốt hơn.
- **Fusion MLP**: nhận vector ghép `2*D` (CNN: 2×1280=2560; Swin: 2×768=1536) → nén về **512 chiều**.
  - `LayerNorm`: ổn định huấn luyện.
  - `SiLU` (Swish): hàm kích hoạt mượt, thường tốt hơn ReLU.
  - `Dropout(0.3)`: chống overfit (dữ liệu y tế ít).
- **Hai đầu ra (multi-task)**:
  - `classification_head` → 1 neuron: logit nhị phân **Normal vs Pathological** (nhiệm vụ chính).
  - `regression_head` → 1 neuron: **dự đoán tuổi** (nhiệm vụ phụ trợ — *auxiliary task* giúp backbone trích đặc trưng giàu hơn, regularize mô hình).

### 4.2 Khởi tạo trọng số `_init_new_layers`
Chỉ các tầng **mới** (fusion + 2 heads) được khởi tạo lại bằng **Kaiming/He uniform**; backbone giữ nguyên trọng số pretrained. Đây là chuẩn mực fine-tuning: không đụng phần đã học tốt, chỉ random phần mới thêm.

### 4.3 `forward` — Lan truyền tiến (4 bước)
```python
# Bước 1 — Trích đặc trưng 2 mắt (qua CÙNG backbone)
left_feat  = self.backbone(left_image)    # [B, D]
right_feat = self.backbone(right_image)   # [B, D]

# Bước 2 — Che mắt bị thiếu (masking)
left_mask  = (~left_missing).float().unsqueeze(1)   # [B, 1]
right_mask = (~right_missing).float().unsqueeze(1)
left_feat  = left_feat  * left_mask                 # mắt thiếu → vector 0
right_feat = right_feat * right_mask

# Bước 3 — Ghép nối + Fusion
fused = torch.cat([left_feat, right_feat], dim=-1)  # [B, 2D]
fused = self.fusion_mlp(fused)                       # [B, 512]

# Bước 4 — Hai nhánh đầu ra (luôn trả về dict)
return {
    "logits":   self.classification_head(fused),  # [B, 1]
    "age_pred": self.regression_head(fused),      # [B, 1]
}
```

> **Bước 2 là điểm thông minh nhất.** Trong thực tế nhiều bệnh nhân chỉ có ảnh 1 mắt; mắt thiếu thường được thay bằng **ảnh đen**. Nếu để ảnh đen đi qua backbone, nó vẫn sinh vector đặc trưng "rác" gây nhiễu. Giải pháp: dùng cờ `left_missing/right_missing` để **nhân đặc trưng mắt thiếu với 0**, loại sạch nhiễu trước khi ghép.
> (`~` đảo bit: mắt *không* thiếu → mask=1, giữ nguyên; mắt thiếu → mask=0, xóa.)

> **Đầu ra luôn là `dict`** chứ không phải tensor đơn — vì có 2 nhiệm vụ. Engine huấn luyện lấy `logits` cho loss chính và `age_pred` cho loss phụ.

### 4.4 Two-stage training: `freeze_backbone` / `unfreeze_backbone`
- **Stage 1** — `freeze_backbone()`: đóng băng backbone (`requires_grad=False`), **chỉ huấn luyện** fusion + 2 heads. Cho các tầng mới "khởi động" mà không phá trọng số pretrained.
- **Stage 2** — `unfreeze_backbone()`: mở khóa toàn mạng, fine-tune với learning rate nhỏ để tinh chỉnh đặc trưng cho riêng ảnh đáy mắt.

Kỹ thuật chuẩn chống "sốc gradient" làm hỏng pretrained ở những epoch đầu.

---

## 5. `__init__.py` — Cổng vào `build_model`
Lớp bọc gọn để các file khác không cần biết chi tiết:
```python
from src.models import build_model
model = build_model(model_type="cnn", img_size=384)   # hoặc "swin"
```
Toàn bộ việc đổi giữa CNN và Swin gói gọn trong tham số `model_type`.

---

## 6. Sơ đồ kiến trúc tổng thể

```
   Ảnh mắt trái [B,3,384,384]      Ảnh mắt phải [B,3,384,384]
            │                               │
            ▼                               ▼
     ┌──────────────  BACKBONE CHIA SẺ TRỌNG SỐ  ──────────────┐
     │  CNN: EfficientNet-B0 (D=1280)  HOẶC  Swin-Tiny (D=768) │
     └─────────────────────────────────────────────────────────┘
            │                               │
       left_feat [B,D]                 right_feat [B,D]
            │  × mask (mắt thiếu→0)         │  × mask
            └──────────────┬────────────────┘
                           ▼
                   concat → [B, 2D]
                           ▼
          Fusion MLP (Linear→LayerNorm→SiLU→Dropout) → [B,512]
                           ▼
            ┌──────────────┴──────────────┐
            ▼                              ▼
   classification_head            regression_head
      logits [B,1]                  age_pred [B,1]
   (Normal/Pathological)             (tuổi - phụ trợ)
```

---

## 7. Luồng hoạt động chi tiết (forward pass)

Theo dấu một batch dữ liệu chạy qua mạng, kèm shape tensor ở từng bước (ví dụ batch size B, ảnh 384×384, backbone CNN D=1280):

```
INPUT (từ DataLoader):
   left_image     [B, 3, 384, 384]
   right_image    [B, 3, 384, 384]
   left_missing   [B]   (bool)
   right_missing  [B]   (bool)

┌─ BƯỚC 1: Trích đặc trưng ──────────────────────────────┐
│  left_image  ──► backbone ──► left_feat   [B, 1280]    │
│  right_image ──► backbone ──► right_feat  [B, 1280]    │
│  (cùng một backbone, chia sẻ trọng số)                 │
└────────────────────────────────────────────────────────┘
                    │
┌─ BƯỚC 2: Masking mắt thiếu ────────────────────────────┐
│  left_mask  = (~left_missing) → [B,1]  (1=có, 0=thiếu) │
│  left_feat  = left_feat  * left_mask   [B, 1280]       │
│  right_feat = right_feat * right_mask  [B, 1280]       │
└────────────────────────────────────────────────────────┘
                    │
┌─ BƯỚC 3: Ghép + Fusion ────────────────────────────────┐
│  concat([left_feat, right_feat]) → fused  [B, 2560]    │
│  fusion_mlp(fused)               → fused  [B, 512]     │
└────────────────────────────────────────────────────────┘
                    │
┌─ BƯỚC 4: Hai đầu ra ───────────────────────────────────┐
│  logits   = classification_head(fused)  [B, 1]         │
│  age_pred = regression_head(fused)      [B, 1]         │
└────────────────────────────────────────────────────────┘
                    │
OUTPUT:  {"logits": [B,1], "age_pred": [B,1]}
                    │
   ┌────────────────┴───────────────────┐
   ▼                                     ▼
 logits → sigmoid → xác suất bệnh    age_pred → tính loss tuổi (phụ)
   (ngưỡng 0.5 → Normal/Pathological)
```

> Với Swin chỉ khác con số `D = 768` (nên `2D = 1536`); cấu trúc luồng giữ nguyên.

---

## 8. Luồng huấn luyện 2 giai đoạn (two-stage)

```
            ┌──────────────────  STAGE 1  ──────────────────┐
            │  model.freeze_backbone()                       │
            │  → backbone ĐÓNG BĂNG (requires_grad=False)    │
            │  → chỉ học: fusion_mlp + 2 heads               │
            │  → learning rate lớn hơn, vài epoch đầu        │
            │  Mục đích: "làm nóng" các tầng mới             │
            └────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌──────────────────  STAGE 2  ──────────────────┐
            │  model.unfreeze_backbone()                     │
            │  → mở khóa TOÀN BỘ mạng                         │
            │  → fine-tune với learning rate NHỎ             │
            │  Mục đích: tinh chỉnh đặc trưng cho ảnh đáy mắt │
            └────────────────────────────────────────────────┘
```

Lý do: nếu mở khóa backbone ngay từ đầu khi các head còn random, gradient lớn sẽ "phá" trọng số pretrained quý giá. Stage 1 ổn định head trước, Stage 2 mới tinh chỉnh nhẹ nhàng.

---

## 9. Bảng tham số nhanh

| Tham số `build_model` | Ý nghĩa | Mặc định |
|---|---|---|
| `model_type` | `"cnn"` (EfficientNet-B0) hoặc `"swin"` (Swin-Tiny) | `"cnn"` |
| `pretrained` | Nạp trọng số ImageNet | `True` |
| `img_size`   | Kích thước ảnh (Swin cần để nội suy vị trí) | `384` |
| `dropout`    | Tỷ lệ dropout trong Fusion MLP | `0.3` |

| Đại lượng | CNN | Swin |
|---|---|---|
| `feature_dim` (D) | 1280 | 768 |
| Vector ghép 2 mắt (2D) | 2560 | 1536 |
| Sau Fusion MLP | 512 | 512 |
| Đầu ra `logits` | [B, 1] | [B, 1] |
| Đầu ra `age_pred` | [B, 1] | [B, 1] |

---

## 10. Câu hỏi thường gặp khi bảo vệ

1. **"Siamese ở đâu trong code?"**
   → Một `self.backbone` duy nhất dùng cho cả 2 mắt (gọi 2 lần trong `forward`). Chia sẻ trọng số chính là bản chất Siamese.

2. **"Vì sao Swin cần `img_size` mà CNN không?"**
   → Relative Position Bias của Swin phụ thuộc độ phân giải; phải nội suy khi đổi từ 224 sang 384. Tích chập thì bất biến kích thước.

3. **"Xử lý bệnh nhân thiếu 1 mắt thế nào?"**
   → Masking nhân 0 ở Bước 2 của `forward`, loại nhiễu từ ảnh đen.

4. **"Vì sao có nhánh dự đoán tuổi?"**
   → Auxiliary multi-task learning: học thêm tuổi giúp backbone trích đặc trưng tốt hơn, đồng thời regularize mô hình.

5. **"`feature_dim` của 2 model là bao nhiêu?"**
   → CNN 1280, Swin 768 (đọc tự động qua `backbone.num_features`).

6. **"Vì sao dùng `timm` thay vì tự code?"**
   → Chuẩn công nghiệp, có pretrained ImageNet sẵn, ít lỗi, dễ tái lập kết quả.

7. **"Khác biệt cốt lõi CNN vs Swin?"**
   → CNN mạnh đặc trưng cục bộ (bộ lọc trượt); Swin mạnh quan hệ toàn cục (self-attention theo cửa sổ + shifted window).
