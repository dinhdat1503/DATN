# 📊 Báo Cáo Phân Tích Mối Quan Hệ Tuổi & Bệnh Lý Võng Mạc (ODIR-5K)

> Phân tích trên tổng số **6364 mẫu** (đã gộp các tập Train, Val, Test và lọc bỏ tuổi < 5).

---
## 1. Thống Kê Độ Tuổi Theo Từng Lớp Bệnh (Ages per Disease)

| Label | Tên bệnh                              | Số ca | Tuổi TB | Tuổi Min | Tuổi 25% | Tuổi Median | Tuổi 75% | Tuổi Max |
|-------|---------------------------------------|-------|---------|----------|----------|-------------|----------|----------|
| N     | Normal (Bình thường)                  | 2101  | 56.7    | 14       | 50.0     | 57.0        | 64.0     | 89       |
| D     | Diabetes Retinopathy (Tiểu đường)     | 2123  | 56.3    | 17       | 50.0     | 57.0        | 64.0     | 85       |
| G     | Glaucoma (Tăng nhãn áp)               | 397   | 62.4    | 24       | 54.0     | 64.0        | 70.0     | 84       |
| C     | Cataract (Đục thủy tinh thể)          | 402   | 66.5    | 24       | 59.0     | 67.0        | 75.0     | 91       |
| A     | AMD (Thoái hóa điểm vàng)             | 317   | 61.2    | 32       | 55.0     | 62.0        | 67.0     | 87       |
| H     | Hypertension (Tăng huyết áp)          | 203   | 56.1    | 34       | 51.0     | 58.0        | 62.0     | 70       |
| M     | Pathological Myopia (Cận thị bệnh lý) | 281   | 61.5    | 29       | 54.0     | 62.0        | 70.0     | 87       |
| O     | Other Diseases (Bệnh lý khác)         | 1586  | 59.4    | 23       | 54.0     | 60.0        | 66.0     | 89       |

### 💡 Nhận xét chính:
- **👵 Bệnh nhân cao tuổi nhất:** Bệnh **Cataract (Đục thủy tinh thể) (C)** có độ tuổi trung bình cao nhất là **66.5 tuổi** (25% bệnh nhân trên 75.0 tuổi).
- **👦 Bệnh nhân trẻ tuổi nhất:** Nhóm **Hypertension (Tăng huyết áp) (H)** có độ tuổi trung bình trẻ nhất là **56.1 tuổi**.
- Bệnh lý thoái hóa điểm vàng (**A - AMD**) và đục thủy tinh thể (**C - Cataract**) hầu như chỉ xảy ra từ độ tuổi trung niên trở lên (Tuổi tối thiểu lần lượt là 35 và 26, tuổi 25% là khoảng trên 56 tuổi).

---
## 2. Tần Suất Mắc Bệnh Theo Từng Nhóm Tuổi (Disease Prevalence by Age Group)

> Bảng dưới đây thể hiện số ca mắc và tỷ lệ % mắc bệnh trong từng nhóm tuổi (Lưu ý: Một bệnh nhân có thể mắc nhiều bệnh).

| Nhóm tuổi    | Tổng số mẫu | N (%) | D (%) | G (%) | C (%) | A (%) | H (%) | M (%) | O (%) |
|--------------|-------------|-------|-------|-------|-------|-------|-------|-------|-------|
| < 40 tuổi    | 346         | 39.6% | 35.5% | 5.8%  | 3.5%  | 2.3%  | 2.3%  | 3.8%  | 19.7% |
| 40 - 49 tuổi | 984         | 37.1% | 38.7% | 3.7%  | 1.8%  | 3.2%  | 3.7%  | 2.1%  | 20.7% |
| 50 - 59 tuổi | 2017        | 35.0% | 38.1% | 4.1%  | 3.6%  | 5.0%  | 3.7%  | 4.0%  | 22.5% |
| 60 - 69 tuổi | 2117        | 30.9% | 32.4% | 6.7%  | 5.9%  | 5.8%  | 3.9%  | 4.3%  | 28.6% |
| >= 70 tuổi   | 900         | 26.6% | 18.4% | 13.0% | 19.3% | 6.1%  | 0.2%  | 8.3%  | 28.3% |

### 💡 Phân tích chi tiết theo nhóm tuổi:

### 📍 Nhóm < 40 tuổi (Tổng cộng 346 mẫu):
- **Normal (Bình thường) (N)**: 137 ca (39.6%)
- **Diabetes Retinopathy (Tiểu đường) (D)**: 123 ca (35.5%)
- **Other Diseases (Bệnh lý khác) (O)**: 68 ca (19.7%)
- **Glaucoma (Tăng nhãn áp) (G)**: 20 ca (5.8%)
- **Pathological Myopia (Cận thị bệnh lý) (M)**: 13 ca (3.8%)
- **Cataract (Đục thủy tinh thể) (C)**: 12 ca (3.5%)
- **AMD (Thoái hóa điểm vàng) (A)**: 8 ca (2.3%)
- **Hypertension (Tăng huyết áp) (H)**: 8 ca (2.3%)

### 📍 Nhóm 40 - 49 tuổi (Tổng cộng 984 mẫu):
- **Diabetes Retinopathy (Tiểu đường) (D)**: 381 ca (38.7%)
- **Normal (Bình thường) (N)**: 365 ca (37.1%)
- **Other Diseases (Bệnh lý khác) (O)**: 204 ca (20.7%)
- **Glaucoma (Tăng nhãn áp) (G)**: 36 ca (3.7%)
- **Hypertension (Tăng huyết áp) (H)**: 36 ca (3.7%)
- **AMD (Thoái hóa điểm vàng) (A)**: 31 ca (3.2%)
- **Pathological Myopia (Cận thị bệnh lý) (M)**: 21 ca (2.1%)
- **Cataract (Đục thủy tinh thể) (C)**: 18 ca (1.8%)

### 📍 Nhóm 50 - 59 tuổi (Tổng cộng 2017 mẫu):
- **Diabetes Retinopathy (Tiểu đường) (D)**: 768 ca (38.1%)
- **Normal (Bình thường) (N)**: 706 ca (35.0%)
- **Other Diseases (Bệnh lý khác) (O)**: 454 ca (22.5%)
- **AMD (Thoái hóa điểm vàng) (A)**: 100 ca (5.0%)
- **Glaucoma (Tăng nhãn áp) (G)**: 83 ca (4.1%)
- **Pathological Myopia (Cận thị bệnh lý) (M)**: 80 ca (4.0%)
- **Hypertension (Tăng huyết áp) (H)**: 74 ca (3.7%)
- **Cataract (Đục thủy tinh thể) (C)**: 73 ca (3.6%)

### 📍 Nhóm 60 - 69 tuổi (Tổng cộng 2117 mẫu):
- **Diabetes Retinopathy (Tiểu đường) (D)**: 685 ca (32.4%)
- **Normal (Bình thường) (N)**: 654 ca (30.9%)
- **Other Diseases (Bệnh lý khác) (O)**: 605 ca (28.6%)
- **Glaucoma (Tăng nhãn áp) (G)**: 141 ca (6.7%)
- **Cataract (Đục thủy tinh thể) (C)**: 125 ca (5.9%)
- **AMD (Thoái hóa điểm vàng) (A)**: 123 ca (5.8%)
- **Pathological Myopia (Cận thị bệnh lý) (M)**: 92 ca (4.3%)
- **Hypertension (Tăng huyết áp) (H)**: 83 ca (3.9%)

### 📍 Nhóm >= 70 tuổi (Tổng cộng 900 mẫu):
- **Other Diseases (Bệnh lý khác) (O)**: 255 ca (28.3%)
- **Normal (Bình thường) (N)**: 239 ca (26.6%)
- **Cataract (Đục thủy tinh thể) (C)**: 174 ca (19.3%)
- **Diabetes Retinopathy (Tiểu đường) (D)**: 166 ca (18.4%)
- **Glaucoma (Tăng nhãn áp) (G)**: 117 ca (13.0%)
- **Pathological Myopia (Cận thị bệnh lý) (M)**: 75 ca (8.3%)
- **AMD (Thoái hóa điểm vàng) (A)**: 55 ca (6.1%)
- **Hypertension (Tăng huyết áp) (H)**: 2 ca (0.2%)
