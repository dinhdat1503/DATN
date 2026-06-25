# Lý thuyết nền tảng: CNN & Swin Transformer (giải thích để tự học)

> Tài liệu học lý thuyết từ con số 0, viết cho người **chưa biết** về mạng nơ-ron, bám sát đúng bài toán đồ án: **phân loại ảnh đáy mắt ODIR-5K (Normal vs Pathological) bằng mạng Siamese 2 mắt**.
>
> Bộ tài liệu liên quan:
> - Kiến trúc code model: [`GIAI_THICH_MO_HINH_CNN_SWIN.md`](GIAI_THICH_MO_HINH_CNN_SWIN.md)
> - Code huấn luyện: [`GIAI_THICH_CODE_HUAN_LUYEN_CNN_SWIN.md`](GIAI_THICH_CODE_HUAN_LUYEN_CNN_SWIN.md)

## Cách dùng tài liệu này
Đọc theo thứ tự Phần 0 → 3. Mỗi phần đi từ **trực giác (ví dụ đời thường)** rồi mới tới **thuật ngữ kỹ thuật**. Cuối tài liệu có **bảng thuật ngữ** và **câu hỏi tự kiểm tra** — hãy tự trả lời trước khi xem lại.

---

## Mục lục
- [Phần 0 — Máy tính "nhìn" ảnh như thế nào](#phần-0--máy-tính-nhìn-ảnh-như-thế-nào)
- [Phần 1 — Mạng nơ-ron tích chập (CNN)](#phần-1--mạng-nơ-ron-tích-chập-cnn)
- [Phần 2 — Swin Transformer](#phần-2--swin-transformer)
- [Phần 3 — Áp dụng vào bài toán của bạn](#phần-3--áp-dụng-vào-bài-toán-của-bạn)
- [Bảng thuật ngữ Việt – Anh](#bảng-thuật-ngữ-việt--anh)
- [Câu hỏi tự kiểm tra](#câu-hỏi-tự-kiểm-tra)
- [Lộ trình học gợi ý](#lộ-trình-học-gợi-ý)

---

## Phần 0 — Máy tính "nhìn" ảnh như thế nào

Máy tính không "thấy" ảnh như con người. Với máy, **một ảnh chỉ là một lưới các con số (pixel)** — mỗi số là độ sáng/màu. Ảnh đáy mắt 384×384 của bạn = một khối **384 × 384 × 3** con số (3 là 3 kênh màu Đỏ–Lục–Lam).

Bài toán của bạn thực chất là **nhận dạng mẫu hình (pattern recognition)**:
> Từ khối số đó, tìm những "mẫu hình" báo hiệu bệnh (xuất huyết, phù hoàng điểm, mạch máu bất thường…) rồi kết luận **Normal** (khỏe) hay **Pathological** (có bệnh lý).

Không ai lập trình tay được luật "nếu pixel chỗ này tối, chỗ kia sáng thì là bệnh" — quá phức tạp. Vì vậy ta dùng **mạng nơ-ron học sâu** để **tự học** các mẫu hình từ hàng nghìn ảnh có nhãn.

**CNN** và **Swin Transformer** là **hai kiểu kiến trúc mạng** — hai *cách "nhìn" ảnh* khác nhau để làm việc đó.

---

## Phần 1 — Mạng nơ-ron tích chập (CNN)

> CNN = **C**onvolutional **N**eural **N**etwork = Mạng nơ-ron tích chập.

### 1.1 Ý tưởng cốt lõi: "bộ lọc trượt" (tích chập)

Hãy tưởng tượng bạn có một **chiếc kính lúp nhỏ 3×3** gọi là **bộ lọc (filter / kernel)**. Bạn **trượt** nó khắp ảnh, từng vùng nhỏ một. Tại mỗi vị trí, kính lúp "chấm điểm": *vùng này có giống mẫu tôi đang tìm không?*

- Một bộ lọc có thể chuyên dò **cạnh dọc**.
- Bộ lọc khác dò **cạnh ngang**, **đốm tròn**, hoặc **vùng đỏ** (như xuất huyết)…

Phép "trượt kính lúp và chấm điểm" đó chính là **tích chập (convolution)** — nguồn gốc cái tên CNN.

Kết quả khi trượt một bộ lọc khắp ảnh là một **bản đồ đặc trưng (feature map)**: chỗ nào sáng = chỗ đó xuất hiện mẫu hình mà bộ lọc tìm.

> 🔑 **Điểm mấu chốt:** mạng **không được lập trình** sẵn bộ lọc nào. Trong quá trình **huấn luyện**, mạng **tự học** ra hàng trăm bộ lọc hữu ích để phân biệt bệnh.

### 1.2 Xếp tầng: từ chi tiết nhỏ → khái niệm lớn

Sức mạnh của CNN đến từ việc **xếp chồng nhiều tầng tích chập**:

```
Tầng 1   → học thứ ĐƠN GIẢN:  cạnh, góc, đốm màu
Tầng 2   → ghép thành KẾT CẤU: mạch máu, vân, bờ viền
Tầng 3   → ghép thành BỘ PHẬN: đĩa thị, hoàng điểm, vùng xuất huyết
Tầng sâu → ghép thành KHÁI NIỆM: "mắt này trông bất thường"
```

Giống cách bạn nhận ra khuôn mặt: cạnh → mắt/mũi/miệng → cả khuôn mặt. CNN nhìn ảnh **từ cục bộ nhỏ, mở rộng dần ra toàn cục**.

### 1.3 Hai phép phụ trợ quan trọng

- **Pooling (gộp/thu nhỏ):** sau vài tầng, thu nhỏ bản đồ đặc trưng (ví dụ lấy giá trị lớn nhất mỗi vùng 2×2). Mục đích: giảm kích thước, giữ thông tin quan trọng, và giúp mạng **bất biến với dịch chuyển nhỏ** (vết bệnh nằm hơi lệch vẫn nhận ra).
- **Hàm kích hoạt phi tuyến (ReLU / SiLU):** đặt sau mỗi tích chập, giúp mạng học được quan hệ **phi tuyến** (thực tế phức tạp, không phải đường thẳng).

### 1.4 Trong đồ án của bạn: EfficientNet-B0

CNN bạn dùng tên là **EfficientNet-B0** — một CNN thiết kế **cân đối tối ưu** giữa độ sâu (số tầng), độ rộng (số bộ lọc) và độ phân giải ảnh; cho độ chính xác cao mà **nhẹ** (~5 triệu tham số), hợp chạy GPU Kaggle. Cuối cùng nó "vắt" mỗi ảnh mắt thành **một vector 1280 chiều** — bản tóm tắt đặc trưng của con mắt đó.

> ✅ **Thế mạnh CNN:** rất giỏi bắt **đặc trưng cục bộ** (texture, đốm, cạnh) — chính là dấu hiệu tổn thương võng mạc. Nhẹ, nhanh, học ổn định kể cả khi dữ liệu vừa phải.

---

## Phần 2 — Swin Transformer

Swin đi theo triết lý **khác hẳn CNN**. Muốn hiểu, cần nắm khái niệm **attention (cơ chế chú ý)** trước.

### 2.1 Attention (sự chú ý) — trực giác

Transformer vốn sinh ra cho **xử lý ngôn ngữ**. Ý tưởng: khi hiểu một từ, ta nên "chú ý" tới các từ liên quan. Ví dụ câu *"con mèo đuổi con chuột vì **nó** đói"* — để hiểu "nó" là ai, mô hình phải **nhìn lại** và chú ý nhiều vào "con mèo".

**Self-attention** = mỗi phần tử (từ, hoặc vùng ảnh) **tự nhìn tất cả phần tử khác** và quyết định "nên chú ý vào ai, bao nhiêu". Đây là cách nắm bắt **quan hệ ở khoảng cách xa**.

### 2.2 Áp dụng attention cho ảnh: chia "patch"

Để dùng attention cho ảnh, ta **chia ảnh thành các ô vuông nhỏ (patch)**, ví dụ 4×4 pixel, rồi coi mỗi patch như một "từ". Ảnh trở thành một "câu" gồm nhiều patch; self-attention cho phép mỗi patch nhìn các patch khác.

**Vấn đề:** nếu mỗi patch nhìn *toàn bộ* patch khác, chi phí tính toán tăng theo **bình phương** số patch → ảnh lớn cực kỳ tốn kém. Đó là lý do Transformer thuần (ViT) khó dùng cho ảnh độ phân giải cao.

### 2.3 Cái hay của Swin: "cửa sổ trượt" + "phân cấp"

> Swin = **S**hifted **WIN**dow Transformer = Transformer cửa sổ dịch chuyển.

Swin giải quyết vấn đề trên bằng 3 ý tưởng:

**(a) Window attention — chú ý trong cửa sổ:** thay vì mỗi patch nhìn *tất cả*, Swin chỉ cho các patch **trong cùng một cửa sổ nhỏ (7×7 patch)** nhìn nhau → rẻ hơn rất nhiều.

**(b) Shifted window — dịch cửa sổ:** nếu cửa sổ cố định, thông tin không "rò" qua biên. Nên tầng kế tiếp Swin **dịch lưới cửa sổ đi nửa bước**, làm patch ở rìa cửa sổ cũ chung cửa sổ mới với hàng xóm → thông tin lan dần ra toàn ảnh qua nhiều tầng.

```
Tầng A: cửa sổ cố định      Tầng B: cửa sổ DỊCH đi nửa bước
┌───┬───┐                   ┌─┬───┬─┐
│ ▢ │ ▢ │   ── dịch ──►     │ │ ▢ │ │   thông tin "rò"
├───┼───┤                   ├─┼───┼─┤   sang cửa sổ lân cận
│ ▢ │ ▢ │                   │ │ ▢ │ │
└───┴───┘                   └─┴───┴─┘
```

**(c) Phân cấp (hierarchical):** giống pooling của CNN, Swin **gộp patch (patch merging)** sau mỗi giai đoạn — lưới patch nhỏ dần, mỗi patch phủ vùng rộng hơn, đặc trưng trừu tượng hơn. Nhờ vậy Swin cũng đi từ **chi tiết → tổng quát** như CNN.

### 2.4 Trong đồ án của bạn: Swin-Tiny

Bạn dùng **Swin-Tiny** (bản nhỏ nhất). Nó nhận ảnh 384×384, chia patch, qua 4 giai đoạn (window-attention + gộp patch), cuối cùng "vắt" mỗi ảnh mắt thành **một vector 768 chiều** — vai trò y hệt vector 1280 của CNN, chỉ "nhìn" theo cách khác.

> ✅ **Thế mạnh Swin:** giỏi nắm **quan hệ toàn cục** và phụ thuộc xa (ví dụ liên hệ giữa đĩa thị và tổn thương ở rìa ảnh) nhờ attention.

### 2.5 So sánh nhanh CNN vs Swin

| | CNN (EfficientNet-B0) | Swin Transformer (Tiny) |
|---|---|---|
| Cách nhìn ảnh | Bộ lọc trượt cục bộ | Patch + chú ý theo cửa sổ |
| Thế mạnh | Đặc trưng cục bộ (đốm, texture) | Quan hệ toàn cục, phụ thuộc xa |
| Nhu cầu dữ liệu | Ít hơn, học ổn định | "Khát" dữ liệu hơn → pretrained rất cần |
| Chi phí tính toán | Nhẹ, nhanh | Nặng hơn (tốn VRAM) |
| Vector đặc trưng | 1280 chiều | 768 chiều |

> ⚖️ Không kiến trúc nào "luôn thắng". **Đó là lý do đồ án thử CẢ HAI** — để xem trên dữ liệu ODIR-5K kiến trúc nào tốt hơn. Đây là một đóng góp khoa học hợp lý của đồ án.

---

## Phần 3 — Áp dụng vào bài toán của bạn

CNN/Swin **không tự nó** giải bài toán. Chúng chỉ là **bộ trích đặc trưng (backbone)** — phần "đôi mắt" của hệ thống. Quanh nó bạn lắp thêm phần "bộ não ra quyết định".

### 3.1 Vì sao cần Siamese (2 mắt)?

Bệnh nhân ODIR-5K có **2 ảnh: mắt trái + mắt phải**, và chẩn đoán là cho **cả người**, không phải từng mắt riêng. Nên kiến trúc là:

```
 Ảnh mắt TRÁI ─►┐
                ├─► CÙNG MỘT backbone (CNN hoặc Swin) ─► 2 vector đặc trưng
 Ảnh mắt PHẢI ─►┘            (chia sẻ trọng số)
                                     │
                          ghép 2 vector lại
                                     │
                        bộ não nhỏ (Fusion MLP)
                                     │
                          Normal  /  Pathological
```

- **"Cùng một backbone cho cả 2 mắt"** = ý tưởng **Siamese (mạng Xiêm)**: dùng chung bộ tham số để học một "cách đọc ảnh mắt" thống nhất → tiết kiệm tham số, tổng quát tốt.
- Backbone (CNN/Swin) biến mỗi mắt → 1 vector đặc trưng.
- **Fusion MLP** gộp đặc trưng 2 mắt → đưa ra **1 con số**: xác suất bệnh. Nếu > ngưỡng (0.5 hoặc ngưỡng Youden) → Pathological.
- Có thêm một **nhánh phụ đoán tuổi** để "ép" backbone học đặc trưng giàu hơn (nhiệm vụ phụ trợ — *auxiliary task*).

### 3.2 Transfer learning — vì sao dùng "pretrained ImageNet"?

Dữ liệu y tế ít (vài nghìn ảnh), không đủ để dạy mạng nhìn từ con số 0. Giải pháp: lấy CNN/Swin **đã học sẵn trên ImageNet** (1,2 triệu ảnh đời thường) — chúng **đã biết** nhìn cạnh, texture, hình khối. Bạn chỉ **tinh chỉnh (fine-tune)** lại cho ảnh đáy mắt.

→ Đây là lý do kết quả tốt hơn nhiều so với học từ đầu, và là lý do có kỹ thuật **two-stage** (đóng băng backbone → mở khóa) trong code huấn luyện.

### 3.3 Toàn cảnh: backbone chỉ là một mắt xích

```
Ảnh thô → [Tiền xử lý: crop, CLAHE] → [Backbone CNN/Swin: trích đặc trưng]
        → [Siamese gộp 2 mắt] → [Phân loại Normal/Pathological]
        → [Huấn luyện: Focal Loss, two-stage, early stopping]
        → [Đánh giá: AUC, Sensitivity, Specificity]
```

CNN và Swin **chỉ thay nhau ở ô "Backbone"**. Mọi thứ khác giữ nguyên — vì vậy trong code, đổi mô hình chỉ là đổi `model_type: cnn ↔ swin`.

---

## Bảng thuật ngữ Việt – Anh

| Tiếng Việt | Tiếng Anh | Nghĩa ngắn gọn |
|---|---|---|
| Mạng nơ-ron tích chập | Convolutional Neural Network (CNN) | Mạng dùng bộ lọc trượt để dò mẫu trong ảnh |
| Tích chập | Convolution | Phép "trượt bộ lọc + chấm điểm" khắp ảnh |
| Bộ lọc / nhân | Filter / Kernel | "Kính lúp" nhỏ dò một loại mẫu hình |
| Bản đồ đặc trưng | Feature map | Kết quả sau khi áp một bộ lọc lên ảnh |
| Gộp / thu nhỏ | Pooling | Thu nhỏ bản đồ đặc trưng, giữ thông tin chính |
| Hàm kích hoạt | Activation (ReLU/SiLU) | Thêm tính phi tuyến cho mạng |
| Cơ chế chú ý | Attention | Mỗi phần tử "chú ý" tới các phần tử liên quan |
| Tự chú ý | Self-attention | Các phần tử trong cùng đầu vào chú ý lẫn nhau |
| Ô ảnh | Patch | Một ô vuông nhỏ cắt từ ảnh, coi như "1 từ" |
| Cửa sổ dịch | Shifted window | Dịch lưới cửa sổ để thông tin lan ra toàn ảnh |
| Gộp ô | Patch merging | Gộp patch để mạng phân cấp như pooling |
| Trích đặc trưng | Backbone / Feature extractor | Phần mạng biến ảnh thành vector đặc trưng |
| Học chuyển giao | Transfer learning | Dùng lại trọng số đã học từ tập dữ liệu lớn |
| Tinh chỉnh | Fine-tuning | Huấn luyện tiếp mô hình pretrained cho bài toán mới |
| Mạng Xiêm | Siamese network | Hai nhánh dùng chung trọng số (ở đây: 2 mắt) |
| Nhiệm vụ phụ trợ | Auxiliary task | Nhiệm vụ phụ (đoán tuổi) hỗ trợ học đặc trưng |

---

## Câu hỏi tự kiểm tra

Hãy tự trả lời, rồi đối chiếu với nội dung phía trên.

1. Với máy tính, một bức ảnh thực chất là gì?
2. "Tích chập" trong CNN nghĩa là làm gì? Tại sao gọi là "bộ lọc trượt"?
3. Vì sao CNN phải **xếp nhiều tầng**? Mỗi tầng học cái gì khác nhau?
4. Pooling dùng để làm gì?
5. Self-attention khác phép tích chập ở điểm cốt lõi nào?
6. Swin giải quyết vấn đề "attention quá tốn kém trên ảnh" bằng 2 ý tưởng nào?
7. "Shifted window" giải quyết hạn chế gì của "window attention"?
8. CNN và Swin, cái nào giỏi đặc trưng **cục bộ**, cái nào giỏi quan hệ **toàn cục**?
9. Backbone (CNN/Swin) cho ra cái gì? Kích thước vector của mỗi loại là bao nhiêu?
10. "Siamese" trong đồ án nghĩa là gì? Vì sao cần nó cho ODIR-5K?
11. Vì sao dùng pretrained ImageNet thay vì học từ đầu?
12. Trong code, để chuyển giữa CNN và Swin ta đổi gì?

<details>
<summary>Gợi ý đáp án ngắn (bấm xem)</summary>

1. Một lưới các con số pixel (384×384×3).
2. Trượt một bộ lọc nhỏ khắp ảnh và chấm điểm mức khớp mẫu; gọi vậy vì bộ lọc "trượt".
3. Để học từ đơn giản (cạnh) → phức tạp (khái niệm); tầng nông học chi tiết, tầng sâu học trừu tượng.
4. Thu nhỏ kích thước, giữ thông tin chính, bất biến dịch chuyển nhỏ.
5. Tích chập chỉ nhìn vùng cục bộ quanh điểm; self-attention cho mỗi phần tử nhìn (gần như) mọi phần tử khác → quan hệ xa.
6. Window attention (chỉ attention trong cửa sổ nhỏ) + shifted window (dịch cửa sổ).
7. Thông tin không "rò" qua biên cửa sổ cố định; dịch cửa sổ để các vùng lân cận giao tiếp.
8. CNN giỏi cục bộ; Swin giỏi toàn cục.
9. Một vector đặc trưng; CNN 1280-D, Swin 768-D.
10. Dùng chung một backbone cho cả 2 mắt (chia sẻ trọng số); vì chẩn đoán cho cả bệnh nhân có 2 ảnh mắt.
11. Dữ liệu y tế ít; pretrained đã biết nhìn cạnh/texture nên fine-tune hiệu quả hơn nhiều.
12. Đổi `model_type: cnn ↔ swin` trong file config.
</details>

---

## Lộ trình học gợi ý

1. **Đọc tài liệu này** (Phần 0→3) cho tới khi trả lời được hết 12 câu tự kiểm tra.
2. Mở [`GIAI_THICH_MO_HINH_CNN_SWIN.md`](GIAI_THICH_MO_HINH_CNN_SWIN.md) — xem lý thuyết hiện ra trong **code model** thực tế (backbone, Siamese forward).
3. Mở [`GIAI_THICH_CODE_HUAN_LUYEN_CNN_SWIN.md`](GIAI_THICH_CODE_HUAN_LUYEN_CNN_SWIN.md) — hiểu mạng được **huấn luyện** ra sao (loss, two-stage, early stopping).
4. Chạy thử `python train.py --config configs/exp_3_cnn_binary_enhanced_aug.yaml --dry-run` và đối chiếu từng bước với tài liệu.

> 💡 Khi bảo vệ, hãy luôn nối lý thuyết về **bài toán cụ thể**: "CNN/Swin là đôi mắt trích đặc trưng; Siamese ghép 2 mắt; tôi thử cả hai để so sánh kiến trúc nào hợp dữ liệu ODIR-5K hơn."
