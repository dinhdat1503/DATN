# 🔍 Báo Cáo Confusion Analysis — CNN

> Phân tích trên **954 mẫu** tập Test với ngưỡng θ=0.5

---
## 1. F1-Score Theo Từng Bệnh (Per-Label Metrics)

| Bệnh                                 | Support | TP  | FP  | FN  | TN  | Precision | Recall | F1     |
|--------------------------------------|---------|-----|-----|-----|-----|-----------|--------|--------|
| N (Normal)                           | 311     | 238 | 269 | 73  | 374 | 0.4694    | 0.7653 | 0.5819 |
| D (Diabetes Retinopathy)             | 317     | 196 | 146 | 121 | 491 | 0.5731    | 0.6183 | 0.5948 |
| G (Glaucoma)                         | 58      | 21  | 49  | 37  | 847 | 0.3       | 0.3621 | 0.3281 |
| C (Cataract)                         | 62      | 52  | 82  | 10  | 810 | 0.3881    | 0.8387 | 0.5306 |
| A (Age-related Macular Degeneration) | 62      | 33  | 37  | 29  | 855 | 0.4714    | 0.5323 | 0.5    |
| H (Hypertension Retinopathy)         | 43      | 5   | 5   | 38  | 906 | 0.5       | 0.1163 | 0.1887 |
| M (Pathological Myopia)              | 41      | 36  | 29  | 5   | 884 | 0.5538    | 0.878  | 0.6792 |
| O (Other Diseases)                   | 225     | 77  | 153 | 148 | 576 | 0.3348    | 0.3422 | 0.3385 |

- **🔴 Bệnh yếu nhất:** H (Hypertension Retinopathy) — F1 = **0.1887** (FN=38, FP=5)
- **🟢 Bệnh mạnh nhất:** M (Pathological Myopia) — F1 = **0.6792**

---
## 2. Ma Trận Nhầm Lẫn Đa Nhãn (Multi-label Confusion Matrix)

> Đọc theo **hàng**: Khi model dự đoán **sai** label ở hàng → bao nhiêu mẫu trong đó thực tế **có** label ở cột?
> Đường chéo = tổng số mẫu bị sai trên chính label đó.

|             | N   | D   | G  | C  | A  | H  | M  | O   |
|-------------|-----|-----|----|----|----|----|----|-----|
| N (thực tế) | 342 | 128 | 32 | 9  | 23 | 11 | 4  | 106 |
| D (thực tế) | 74  | 267 | 9  | 10 | 24 | 14 | 3  | 73  |
| G (thực tế) | 20  | 11  | 86 | 7  | 8  | 6  | 5  | 20  |
| C (thực tế) | 26  | 19  | 16 | 92 | 1  | 1  | 10 | 34  |
| A (thực tế) | 12  | 15  | 5  | 1  | 66 | 2  | 0  | 24  |
| H (thực tế) | 0   | 22  | 5  | 2  | 2  | 43 | 0  | 4   |
| M (thực tế) | 12  | 4   | 5  | 3  | 3  | 1  | 34 | 11  |
| O (thực tế) | 46  | 105 | 13 | 14 | 24 | 28 | 12 | 301 |

---
## 3. Ma Trận Xác Suất Trung Bình (Probability Confusion)

> Đọc: Khi bệnh nhân **thực tế mắc** bệnh ở hàng → model cho xác suất trung bình cho từng bệnh ở cột là bao nhiêu?
> **Xác suất cao ngoài đường chéo** = model hay nhầm 2 bệnh này với nhau.

|              | P(N)   | P(D)   | P(G)   | P(C)   | P(A)   | P(H)   | P(M)   | P(O)   |
|--------------|--------|--------|--------|--------|--------|--------|--------|--------|
| N (thực mắc) | 0.6305 | 0.3732 | 0.2396 | 0.2099 | 0.2237 | 0.1214 | 0.0905 | 0.3595 |
| D (thực mắc) | 0.4204 | 0.5845 | 0.213  | 0.1955 | 0.2291 | 0.1612 | 0.0657 | 0.4525 |
| G (thực mắc) | 0.5256 | 0.3001 | 0.4983 | 0.3791 | 0.2415 | 0.1204 | 0.2296 | 0.3766 |
| C (thực mắc) | 0.3323 | 0.2793 | 0.2826 | 0.7831 | 0.2728 | 0.1233 | 0.1631 | 0.3541 |
| A (thực mắc) | 0.4048 | 0.4104 | 0.2786 | 0.1727 | 0.5154 | 0.1577 | 0.1129 | 0.4626 |
| H (thực mắc) | 0.343  | 0.6328 | 0.2452 | 0.159  | 0.217  | 0.285  | 0.0771 | 0.5195 |
| M (thực mắc) | 0.2049 | 0.1888 | 0.2359 | 0.3834 | 0.1245 | 0.0511 | 0.8516 | 0.3203 |
| O (thực mắc) | 0.4673 | 0.4393 | 0.2665 | 0.2823 | 0.2651 | 0.1285 | 0.1338 | 0.4479 |

---
## 4. False Positive Analysis (Model Đoán CÓ Nhưng SAI)

> Khi model **đoán bệnh nhân CÓ bệnh X nhưng SAI**, bệnh nhân đó thực tế mắc bệnh gì?

### N (Normal) — 269 FP
Bệnh thực tế của 269 ca bị đoán nhầm CÓ N:

- **D** (Diabetes Retinopathy): 128 ca (47.6%)
- **O** (Other Diseases): 106 ca (39.4%)
- **G** (Glaucoma): 32 ca (11.9%)
- **A** (Age-related Macular Degeneration): 23 ca (8.6%)
- **H** (Hypertension Retinopathy): 11 ca (4.1%)
- **C** (Cataract): 9 ca (3.3%)
- **M** (Pathological Myopia): 4 ca (1.5%)

### D (Diabetes Retinopathy) — 146 FP
Bệnh thực tế của 146 ca bị đoán nhầm CÓ D:

- **N** (Normal): 74 ca (50.7%)
- **O** (Other Diseases): 47 ca (32.2%)
- **A** (Age-related Macular Degeneration): 14 ca (9.6%)
- **H** (Hypertension Retinopathy): 12 ca (8.2%)
- **G** (Glaucoma): 4 ca (2.7%)
- **C** (Cataract): 2 ca (1.4%)
- **M** (Pathological Myopia): 1 ca (0.7%)

### G (Glaucoma) — 49 FP
Bệnh thực tế của 49 ca bị đoán nhầm CÓ G:

- **N** (Normal): 20 ca (40.8%)
- **O** (Other Diseases): 16 ca (32.7%)
- **D** (Diabetes Retinopathy): 9 ca (18.4%)
- **C** (Cataract): 7 ca (14.3%)
- **A** (Age-related Macular Degeneration): 3 ca (6.1%)
- **H** (Hypertension Retinopathy): 2 ca (4.1%)
- **M** (Pathological Myopia): 2 ca (4.1%)

### C (Cataract) — 82 FP
Bệnh thực tế của 82 ca bị đoán nhầm CÓ C:

- **O** (Other Diseases): 34 ca (41.5%)
- **N** (Normal): 26 ca (31.7%)
- **G** (Glaucoma): 16 ca (19.5%)
- **D** (Diabetes Retinopathy): 13 ca (15.9%)
- **M** (Pathological Myopia): 10 ca (12.2%)
- **A** (Age-related Macular Degeneration): 1 ca (1.2%)

### A (Age-related Macular Degeneration) — 37 FP
Bệnh thực tế của 37 ca bị đoán nhầm CÓ A:

- **O** (Other Diseases): 22 ca (59.5%)
- **N** (Normal): 12 ca (32.4%)
- **D** (Diabetes Retinopathy): 7 ca (18.9%)
- **G** (Glaucoma): 2 ca (5.4%)
- **C** (Cataract): 1 ca (2.7%)
- **H** (Hypertension Retinopathy): 1 ca (2.7%)

### H (Hypertension Retinopathy) — 5 FP
Bệnh thực tế của 5 ca bị đoán nhầm CÓ H:

- **D** (Diabetes Retinopathy): 3 ca (60.0%)
- **G** (Glaucoma): 1 ca (20.0%)
- **A** (Age-related Macular Degeneration): 1 ca (20.0%)
- **O** (Other Diseases): 1 ca (20.0%)

### M (Pathological Myopia) — 29 FP
Bệnh thực tế của 29 ca bị đoán nhầm CÓ M:

- **N** (Normal): 12 ca (41.4%)
- **O** (Other Diseases): 9 ca (31.0%)
- **G** (Glaucoma): 5 ca (17.2%)
- **D** (Diabetes Retinopathy): 3 ca (10.3%)
- **C** (Cataract): 3 ca (10.3%)
- **A** (Age-related Macular Degeneration): 3 ca (10.3%)
- **H** (Hypertension Retinopathy): 1 ca (3.4%)

### O (Other Diseases) — 153 FP
Bệnh thực tế của 153 ca bị đoán nhầm CÓ O:

- **D** (Diabetes Retinopathy): 76 ca (49.7%)
- **N** (Normal): 46 ca (30.1%)
- **H** (Hypertension Retinopathy): 26 ca (17.0%)
- **A** (Age-related Macular Degeneration): 23 ca (15.0%)
- **G** (Glaucoma): 7 ca (4.6%)
- **C** (Cataract): 7 ca (4.6%)
- **M** (Pathological Myopia): 4 ca (2.6%)

---
## 5. False Negative Analysis (Model BỎ SÓT Bệnh)

> Khi model **bỏ sót bệnh X** (có bệnh nhưng model đoán không), model nghĩ bệnh nhân mắc bệnh gì?

### N (Normal) — 73 FN
Trong 73 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **D** (Diabetes Retinopathy): 52 ca (71.2%)
- **O** (Other Diseases): 38 ca (52.1%)
- **A** (Age-related Macular Degeneration): 7 ca (9.6%)
- **C** (Cataract): 3 ca (4.1%)
- **M** (Pathological Myopia): 2 ca (2.7%)
- **G** (Glaucoma): 1 ca (1.4%)

### D (Diabetes Retinopathy) — 121 FN
Trong 121 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **N** (Normal): 96 ca (79.3%)
- **O** (Other Diseases): 20 ca (16.5%)
- **C** (Cataract): 16 ca (13.2%)
- **G** (Glaucoma): 6 ca (5.0%)
- **A** (Age-related Macular Degeneration): 5 ca (4.1%)
- **M** (Pathological Myopia): 1 ca (0.8%)

### G (Glaucoma) — 37 FN
Trong 37 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **N** (Normal): 19 ca (51.4%)
- **C** (Cataract): 10 ca (27.0%)
- **O** (Other Diseases): 9 ca (24.3%)
- **M** (Pathological Myopia): 7 ca (18.9%)
- **D** (Diabetes Retinopathy): 3 ca (8.1%)
- **A** (Age-related Macular Degeneration): 3 ca (8.1%)
- **H** (Hypertension Retinopathy): 1 ca (2.7%)

### C (Cataract) — 10 FN
Trong 10 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **D** (Diabetes Retinopathy): 8 ca (80.0%)
- **O** (Other Diseases): 7 ca (70.0%)
- **N** (Normal): 2 ca (20.0%)
- **G** (Glaucoma): 1 ca (10.0%)

### A (Age-related Macular Degeneration) — 29 FN
Trong 29 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **N** (Normal): 14 ca (48.3%)
- **O** (Other Diseases): 13 ca (44.8%)
- **D** (Diabetes Retinopathy): 12 ca (41.4%)
- **G** (Glaucoma): 2 ca (6.9%)
- **M** (Pathological Myopia): 2 ca (6.9%)
- **H** (Hypertension Retinopathy): 1 ca (3.4%)
- **C** (Cataract): 1 ca (3.4%)

### H (Hypertension Retinopathy) — 38 FN
Trong 38 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **D** (Diabetes Retinopathy): 28 ca (73.7%)
- **O** (Other Diseases): 23 ca (60.5%)
- **N** (Normal): 11 ca (28.9%)
- **A** (Age-related Macular Degeneration): 2 ca (5.3%)
- **G** (Glaucoma): 1 ca (2.6%)
- **C** (Cataract): 1 ca (2.6%)
- **M** (Pathological Myopia): 1 ca (2.6%)

### M (Pathological Myopia) — 5 FN
Trong 5 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **N** (Normal): 3 ca (60.0%)
- **C** (Cataract): 2 ca (40.0%)
- **D** (Diabetes Retinopathy): 1 ca (20.0%)
- **G** (Glaucoma): 1 ca (20.0%)

### O (Other Diseases) — 148 FN
Trong 148 ca bị bỏ sót, model đoán bệnh nhân mắc:

- **N** (Normal): 97 ca (65.5%)
- **C** (Cataract): 34 ca (23.0%)
- **D** (Diabetes Retinopathy): 33 ca (22.3%)
- **G** (Glaucoma): 17 ca (11.5%)
- **M** (Pathological Myopia): 13 ca (8.8%)
- **A** (Age-related Macular Degeneration): 10 ca (6.8%)

---
## 6. Tóm Tắt & Gợi Ý Cải Thiện F1-Score

### Top cặp bệnh dễ nhầm lẫn nhất (theo xác suất rò rỉ):

1. **H** ↔ **D** (Hypertension Retinopathy ↔ Diabetes Retinopathy): P(D|mắc H) = **0.6328**, P(H|mắc D) = **0.1612**
2. **G** ↔ **N** (Glaucoma ↔ Normal): P(N|mắc G) = **0.5256**, P(G|mắc N) = **0.2396**
3. **H** ↔ **O** (Hypertension Retinopathy ↔ Other Diseases): P(O|mắc H) = **0.5195**, P(H|mắc O) = **0.1285**
4. **O** ↔ **N** (Other Diseases ↔ Normal): P(N|mắc O) = **0.4673**, P(O|mắc N) = **0.3595**
5. **A** ↔ **O** (Age-related Macular Degeneration ↔ Other Diseases): P(O|mắc A) = **0.4626**, P(A|mắc O) = **0.2651**
