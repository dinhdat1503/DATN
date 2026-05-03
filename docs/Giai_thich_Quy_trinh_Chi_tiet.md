# Giải Thích Quy Trình Chi Tiết: Dự Án ODIR-5K Multi-task Learning

Tài liệu này cung cấp một cái nhìn sâu sắc, toàn diện về toàn bộ vòng đời của dự án phân tích ảnh đáy mắt ODIR-5K, từ những pixel thô ban đầu cho đến khi trở thành một hệ thống chẩn đoán AI hoàn chỉnh.

---

## 1. Bài Toán và Mục Tiêu Cốt Lõi

### Bối cảnh
Trong nhãn khoa, ảnh chụp võng mạc (fundus) chứa đựng lượng thông tin khổng lồ về sức khỏe. Không chỉ các bệnh về mắt (như đục thủy tinh thể, tăng nhãn áp) hiển thị trên võng mạc, mà cả các bệnh hệ thống (như tiểu đường, cao huyết áp) cũng làm biến đổi hệ vi mạch máu ở đây. Thêm vào đó, tình trạng võng mạc còn phản ánh mức độ lão hóa sinh học của toàn cơ thể.

### Đầu vào (Input)
- Hình ảnh màu chụp đáy mắt (RGB) từ nhiều nguồn thiết bị, bệnh viện khác nhau.

### Đầu ra (Output) - Bài toán Multi-task
Mô hình AI phải đồng thời thực hiện hai nhiệm vụ:
1. **Phân loại đa nhãn (Multi-label Classification):** Trả về xác suất mắc 8 loại bệnh (N: Bình thường, D: Tiểu đường, G: Tăng nhãn áp, C: Đục thủy tinh thể, A: Thoái hóa điểm vàng, H: Tăng huyết áp, M: Cận thị bệnh lý, O: Các bệnh khác). "Đa nhãn" có nghĩa là một người có thể cùng lúc mắc nhiều bệnh (ví dụ vừa D vừa H).
2. **Hồi quy (Regression):** Dự đoán tuổi sinh học của bệnh nhân dựa trên tình trạng võng mạc. 

Sự chênh lệch giữa *tuổi sinh học dự đoán* và *tuổi thực* được gọi là **Retinal Age Gap** - một dấu ấn sinh học cực kỳ quan trọng để cảnh báo nguy cơ tử vong hoặc các bệnh lý tiềm ẩn.

---

## 2. Tiền Xử Lý Dữ Liệu (Preprocessing) - Chuẩn Bị Tín Hiệu Sạch

AI giống như một đứa trẻ, nếu bạn cho nó nhìn ảnh bẩn, nhiễu, sáng tối khác nhau, nó sẽ học nhầm đặc trưng. Mục tiêu của tiền xử lý là tạo ra sự đồng nhất tuyệt đối.

### Bước 2.1: ROI Cropping (Cắt vùng quan tâm)
- **Vấn đề:** Ảnh chụp võng mạc thường là một vòng tròn sáng ở giữa và khoảng đen rộng lớn xung quanh. Khoảng đen này vô giá trị và làm tốn tài nguyên tính toán.
- **Giải pháp:** Dùng thuật toán phát hiện viền để tự động cắt lấy một khung hình chữ nhật vừa vặn bao quanh nhãn cầu, sau đó nén mọi ảnh về chuẩn `512x512 pixel`.

### Bước 2.2: Ben Graham Normalization (Cân bằng ánh sáng)
- **Vấn đề:** Máy chụp ở bệnh viện A có đèn flash mạnh (ảnh sáng rực), máy ở bệnh viện B có flash yếu (ảnh tối). Nếu đưa vào mô hình, AI có thể tự kết luận: "Cứ ảnh tối là bị bệnh G", vì tình cờ bệnh viện B chuyên khám bệnh G. Đây gọi là *Domain/Device Bias*.
- **Giải pháp:** Áp dụng thuật toán của Ben Graham (người thắng cuộc thi Kaggle về võng mạc). Thuật toán làm mờ ảnh (Gaussian blur) để lấy độ sáng nền, rồi lấy ảnh gốc trừ đi độ sáng nền đó, cuối cùng cộng thêm 128 (màu xám trung tính). Kết quả là mọi ảnh đều có độ sáng đồng đều bất kể máy chụp.

### Bước 2.3: CLAHE (Tương phản thích nghi)
- **Vấn đề:** Các mạch máu nhỏ (vi mạch) hay các nốt xuất huyết li ti rất khó nhìn thấy, chúng thường bị chìm vào nền màu đỏ của võng mạc.
- **Giải pháp:** Chuyển ảnh sang không gian màu LAB (chỉ lấy kênh L - độ sáng). Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization). Khác với cân bằng biểu đồ thông thường làm nhiễu hạt to lên, CLAHE chia ảnh thành lưới `8x8` và cân bằng tương phản trong từng ô nhỏ. Mạch máu sẽ nổi bật lên rõ rệt.

---

## 3. Phân Chia Tập Dữ Liệu (Data Splitting) - Chống Gian Lận

### Vấn đề: Patient Leakage (Rò rỉ dữ liệu bệnh nhân)
Thông thường, chia tập Train (huấn luyện) / Val (kiểm định) / Test (kiểm tra) thường làm bằng cách bốc ngẫu nhiên từng bức ảnh. Tuy nhiên, trong y tế, một bệnh nhân có 2 mắt (2 ảnh). Hai con mắt của cùng một người có chung cấu trúc giải phẫu, gen, mạch máu. 
Nếu mắt trái vào tập Train, mắt phải vào tập Test $\rightarrow$ Mô hình đã được "nhìn lén" đặc trưng của bệnh nhân này lúc học. Khi làm bài kiểm tra (Test), nó đạt điểm cao vì nó nhận ra *người quen*, chứ không phải nhận ra *đặc trưng bệnh*.

### Giải pháp: Phân chia mức độ bệnh nhân (Patient-Level Split)
- Gom ảnh lại theo ID bệnh nhân (mỗi ID có thể có 1 hoặc 2 ảnh).
- Tiến hành chia ID bệnh nhân thành 3 tập (Train: ~70%, Val: ~15%, Test: ~15%).
- Đảm bảo (Stratify) tỉ lệ các bệnh giữa 3 tập là tương đương nhau. 
- *Kết quả:* Không có bất kỳ bệnh nhân nào ở tập Train xuất hiện ở tập Test. Điểm số đánh giá sẽ phản ánh đúng sức mạnh thực tế của AI.

---

## 4. Xử Lý Mất Cân Bằng Dữ Liệu (Class Imbalance) - Chiến Lược Kép

### 4.1. Bản Chất Của Sự Mất Cân Bằng Trong Y Tế
Trong bài toán ODIR-5K, sự phân bố bệnh lý phản ánh đúng thực tế lâm sàng: các ca bình thường (N) hoặc bệnh phổ biến như tiểu đường (D) chiếm đại đa số (mỗi loại khoảng 33% tập dữ liệu). Ngược lại, các ca bệnh nguy hiểm nhưng hiếm gặp như Tăng huyết áp (H) chỉ chiếm vỏn vẹn **3.2%**.

**Hiệu ứng "AI lười biếng" (Overfitting to Majority Class):**
Nếu huấn luyện mô hình bằng phương pháp thông thường (Empirical Risk Minimization), mô hình sẽ rơi vào cạm bẫy tối ưu cục bộ. Nó nhận ra rằng: chỉ cần dự đoán mọi bức ảnh đều là "Bình thường" hoặc "Tiểu đường", nó vẫn đạt độ chính xác (Accuracy) lên tới 66% mà chẳng cần học thuộc các đặc trưng phức tạp của bệnh Tăng huyết áp. Hậu quả là mô hình hoàn toàn **bị mù** trước bệnh nhân tăng huyết áp thực sự (Recall = 0%).

Để giải quyết triệt để, dự án thiết kế một chiến lược bảo vệ kép: Lớp 1 (Trực tiếp) và Lớp 2 (Gián tiếp).

### 4.2. Lớp 1: Trọng Số Hàm Mất Mát (`pos_weight`) - "Bàn Tay Sắt" Định Hướng Học Tập

Khái niệm "trừ điểm" thực chất là một cách nói ẩn dụ cho **Hàm mất mát (Loss Function)** trong Machine Learning. Mục tiêu tối thượng của mô hình AI trong lúc huấn luyện là phải tìm cách thay đổi các tham số nội bộ sao cho giá trị Loss này càng nhỏ càng tốt. Loss càng lớn, mô hình càng bị "phạt" nặng và phải cập nhật trọng số mạnh hơn để sửa lỗi.

**Cơ sở lý thuyết & Công thức toán học:**

Trong bài toán phân loại đa nhãn (Multi-label), hàm mất mát tiêu chuẩn được sử dụng là **BCE (Binary Cross Entropy)**. Công thức BCE gốc cho một nhãn bệnh duy nhất là:

$Loss = - [ y \cdot \log(p) + (1 - y) \cdot \log(1 - p) ]$

*(Trong đó: $y \in \{0, 1\}$ là nhãn thực tế do bác sĩ gán, $p \in (0, 1)$ là xác suất mắc bệnh do AI dự đoán)*

**Vấn đề của BCE gốc trong dữ liệu mất cân bằng:** 
Hãy tưởng tượng có 100 bệnh nhân, trong đó 97 người âm tính ($y=0$) và 3 người dương tính ($y=1$). Tổng Loss của cả batch sẽ bị thống trị bởi vế $(1-y)$ của 97 người âm tính. Mô hình sẽ chọn giải pháp gian lận: luôn dự đoán $p \approx 0$ (Không bệnh) cho tất cả mọi người. Lúc này Loss vẫn rất thấp, nhưng mô hình hoàn toàn vô dụng.

**Giải pháp Cost-Sensitive Learning (Weighted BCE):** 
Thư viện PyTorch cung cấp hàm `BCEWithLogitsLoss` tích hợp sẵn tham số `pos_weight`. Công thức lúc này được biến đổi thành:

$Loss = - [ \mathbf{pos\_weight} \cdot y \cdot \log(p) + (1 - y) \cdot \log(1 - p) ]$

*(Chú ý: hệ số `pos_weight` **chỉ** được nhân vào vế của nhãn dương tính $y=1$)*

Để cân bằng tuyệt đối sức mạnh (đóng góp vào tổng Loss) giữa nhóm đa số và thiểu số, khoa học dữ liệu định nghĩa công thức:
`(Tổng số mẫu dương) × pos_weight = (Tổng số mẫu âm) × 1`
$\rightarrow$ `pos_weight = Tổng số mẫu âm / Tổng số mẫu dương`

*Bảng trọng số thực tế của dự án (từ metadata.json):*
- Nhãn Bình thường (N): 1,478 dương tính, 2,984 âm tính $\rightarrow$ `pos_weight = 2.02`
- **Nhãn Tăng huyết áp (H)**: 132 dương tính, 4,330 âm tính $\rightarrow$ `pos_weight = 32.80`

**Cơ chế hoạt động thực tế:** 
Nhìn vào công thức Weighted BCE, ta thấy:
- Nếu AI bỏ lỡ (dự đoán sai) một bệnh nhân Tăng huyết áp (H) (tức là $y=1$ nhưng $p \approx 0$), giá trị Loss sinh ra từ sai sót đó sẽ bị **nhân lên 32.8 lần**.

**Sự "bùng nổ Loss" này thực chất làm thay đổi AI như thế nào?**

Để hiểu sự cải thiện thực sự diễn ra bên trong mô hình, ta cần nhìn vào thuật toán cốt lõi của Trí tuệ nhân tạo: **Lan truyền ngược (Backpropagation)** và **Giảm dốc (Gradient Descent)**.

1. **Bản chất của sự học:** Mô hình AI (như CNN hay Swin) thực chất là một mạng lưới chứa hàng chục triệu tham số (được gọi là Trọng số - *Weights*). Các trọng số này cấu thành nên các "bộ lọc" (filters) để rà quét và dò tìm các dấu hiệu trên bức ảnh (ví dụ: một bộ lọc chuyên tìm nốt trắng, một bộ lọc chuyên tìm vi mạch máu). Lúc AI mới khởi tạo, các bộ lọc này là ngẫu nhiên và vô dụng.
2. **Vai trò của Gradient (Lực đẩy):** Khi AI dự đoán sai, nó tính ra điểm Loss. Từ giá trị Loss này, bằng giải tích toán học, thuật toán sẽ tính ra một thứ gọi là **Gradient**. Hãy hình dung Gradient chính là *"mũi tên chỉ hướng và lực tác động"* ép các Trọng số phải tự uốn nắn, thay đổi giá trị. Điểm Loss càng lớn $\rightarrow$ Lực Gradient sinh ra càng khổng lồ $\rightarrow$ các Trọng số bị ép thay đổi càng mạnh bạo.
3. **Trước khi có `pos_weight`:** Khi AI đoán sai 1 ca Tăng huyết áp, Loss sinh ra rất nhỏ, chìm lấp trong hàng ngàn ca Bình thường. Lực Gradient sinh ra quá yếu, chỉ như một cái "gãi ngứa". Các "bộ lọc" bên trong AI chọn cách lười biếng, chúng không thèm thay đổi cấu trúc để học cách nhận diện mạch máu của người Tăng huyết áp.
4. **Sau khi bùng nổ Loss (`pos_weight = 32.8`):** Giờ đây, chỉ cần lọt lưới 1 ca Tăng huyết áp, hàm toán học tạo ra một điểm Loss khổng lồ. Lực Gradient sinh ra lúc này như một "cú sốc điện" lan truyền ngược (backpropagate) đánh thẳng vào hàng chục triệu trọng số. Áp lực toán học này **cưỡng ép** các trọng số phải ngay lập tức phá vỡ cấu trúc cũ, tinh chỉnh lại giá trị để biến thành các "bộ lọc" cực kỳ nhạy bén, chuyên săn lùng những tổn thương võng mạc li ti của bệnh Tăng huyết áp nhằm không bị phạt nặng trong các lần học tiếp theo.

Đó chính là cách một phép nhân đơn giản (nhân 32.8 lần Loss) ở đầu ra lại có thể can thiệp sâu sắc vào cấu trúc não bộ của AI, ép nó không được phớt lờ nhóm bệnh nhân thiểu số.

### 4.3. Lớp 2: Tăng Cường Dữ Liệu Nâng Cao (MixUp & CutMix) - "Mở Rộng Tầm Nhìn"
Nếu Lớp 1 dùng "hình phạt" để ép mô hình học, thì Lớp 2 dùng "ảo ảnh" để giúp mô hình thông minh hơn và khái quát hóa tốt hơn (Regularization).

Thay vì chỉ học trên dữ liệu gốc cứng nhắc (Nhãn cứng: bệnh N=1, bệnh H=0), ta tự tạo ra các "bệnh nhân ảo" mang đặc tính lai tạp giữa 2 bệnh nhân thật ở cấp độ Batch (trong quá trình tải dữ liệu). Việc này được chi phối bởi tham số ngẫu nhiên $\lambda$ (Lambda) lấy từ phân phối Beta.

#### A. MixUp (Nội Suy Tuyến Tính Toàn Ảnh)
- **Cách làm:** Lấy ảnh A (Tiểu đường) hòa trộn toàn bộ điểm ảnh (pixel) với ảnh B (Tăng huyết áp) với tỉ lệ $\lambda$, ví dụ $70\%$ A và $30\%$ B.
- **Tuổi & Nhãn bệnh:** Tương ứng, tuổi sẽ được cộng trung bình ($70\%$ tuổi A + $30\%$ tuổi B). Nhãn bệnh trở thành nhãn mềm (Soft-label): `[D: 0.7, H: 0.3]`.
- **Tác dụng:** Buộc mô hình học được một "đường ranh giới quyết định" (Decision boundary) mềm mại, mịn màng hơn thay vì chỉ học vẹt các giá trị nhãn cứng 0 và 1. Tuy nhiên, MixUp có một nhược điểm: việc làm mờ toàn bộ ảnh có thể làm mất đi các cấu trúc cục bộ tinh vi của võng mạc.

#### B. CutMix (Cắt Dán Vùng Cục Bộ)
- **Cách làm:** Phát triển từ MixUp, CutMix không trộn mờ pixel. Thay vào đó, nó **cắt một ô vuông** từ ảnh B và **dán đè** lên ảnh A. Diện tích ô vuông quyết định tỉ lệ $\lambda$ thực tế.
- **Tác dụng vượt trội:** Cực kỳ phù hợp cho ảnh y khoa. Vì hình ảnh võng mạc không bị làm mờ, các cấu trúc cục bộ (như đĩa thị giác, các nốt xuất huyết vi mô) được giữ nguyên vẹn. 
- **Lợi ích với bệnh thiểu số:** Một vết tổn thương của bệnh hiếm (H) giờ đây có thể được "dán" vào bối cảnh nền võng mạc của một bệnh nhân bình thường. Nhờ đó, AI học được cách định vị (localize) đặc trưng bệnh hiếm bất kể nó xuất hiện ở bối cảnh nào.

**Tóm lại:** Nhờ CutMix/MixUp, số lần mô hình "được nhìn thấy" đặc trưng của bệnh Tăng huyết áp đã tăng lên gấp nhiều lần dưới các dạng biến thể pha trộn, gián tiếp giải quyết bài toán thiếu hụt dữ liệu.

---

## 5. Mô Hình AI - Học Đa Nhiệm (Multi-task Learning)

### Tại sao là Multi-task?
Trực giác Y khoa: Tổn thương của các bệnh lý võng mạc thường đi đôi với sự lão hóa mạch máu. Nếu ta bắt AI vừa phải nhận diện bệnh, vừa phải đoán tuổi, các *nơ-ron* bên trong nó sẽ phải chia sẻ thông tin cho nhau. Việc đoán tuổi giúp AI định vị các mạch máu tốt hơn, từ đó làm tăng độ chính xác của việc phân loại bệnh (và ngược lại).

### Cấu trúc mô hình (Hard Parameter Sharing)
- **Backbone (Xương sống):** Phần thân chính chịu trách nhiệm trích xuất đặc trưng hình ảnh. Dự án dùng 2 loại để thi đấu với nhau:
  - *CNN (ResNet/EfficientNet):* Giống như dùng kính lúp soi từng chi tiết cục bộ (vết máu, nốt trắng).
  - *Swin Transformer:* Công nghệ mới, tự phân tích mối liên hệ toàn cục (ví dụ: vết máu góc này có liên quan gì đến mảng trắng góc kia).
- **Heads (Các nhánh đầu ra):** Từ xương sống, mô hình tách làm 2 nhánh rẽ:
  - *Classification Head:* Đi qua hàm Sigmoid để chốt tỉ lệ % cho 8 nhãn bệnh.
  - *Regression Head:* Đầu ra 1 con số duy nhất là tuổi sinh học.

---

## 6. Đánh Giá (Evaluation) & Ablation Study

Làm sao để biết mô hình AI thực sự tốt?
- **Với Bệnh lý:** Không dùng Accuracy (vì mất cân bằng). Dùng **F1-Score** (trung bình hài hòa giữa Precision và Recall) và **AUC-ROC** (khả năng phân định giữa người bệnh và không bệnh).
- **Với Tuổi:** Dùng **MAE** (sai số trung bình tuyệt đối - ví dụ lệch 4 năm) và **Pearson** (sự tương quan tuyến tính giữa tuổi đoán và tuổi thực).

**Ablation Study (Nghiên cứu cắt bỏ):**
Đây là cách làm khoa học để chứng minh lý thuyết. Ta sẽ thử:
- Tắt MixUp/CutMix $\rightarrow$ Xem điểm số bệnh H tụt bao nhiêu?
- Tắt hàm đoán tuổi (chỉ đoán bệnh) $\rightarrow$ Xem điểm số phân loại bệnh có giảm không? (Để chứng minh việc học đa nhiệm là có tác dụng).

---

## 7. Đích Đến Cuối Cùng: Ứng Dụng (Web App)

AI không thể chỉ nằm trong code. Mục đích cuối cùng là cung cấp công cụ cho bác sĩ.
1. Bác sĩ tải ảnh võng mạc từ máy khám lên Web App.
2. Hệ thống chạy Preprocessing (ROI $\rightarrow$ Ben Graham $\rightarrow$ CLAHE).
3. Đưa qua mô hình Swin/CNN tốt nhất.
4. Giao diện hiện ra: 
   - Danh sách các bệnh nghi ngờ (Ví dụ: 85% Tiểu đường, 60% Cao huyết áp).
   - Tuổi thực: 50 | Tuổi võng mạc (AI dự đoán): 58 $\rightarrow$ Retinal Age Gap: +8 tuổi (Cảnh báo lão hóa sớm).
   - **Heatmap (Grad-CAM):** Vẽ một bản đồ nhiệt màu đỏ rực lên chính xác cái khu vực xuất huyết võng mạc để giải thích cho bác sĩ: *"Tôi đoán nó bị tiểu đường là vì vết máu ở chỗ này"*.
