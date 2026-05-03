# ODIR-5K Multi-task Learning — Đồ Án Tốt Nghiệp

**Đề tài**: Nghiên cứu và ứng dụng học sâu hỗ trợ chẩn đoán bệnh lý nhãn khoa và dự đoán tuổi sinh học từ ảnh đáy mắt

**Sinh viên**: Ngô Đình Đạt — MSSV 2251161965 — Lớp 64HTTT2  
**GVHD**: TS. Lê Thị Tú Kiên  
**Trường**: Đại học Thuỷ Lợi — Khoa Công nghệ Thông tin

---

## Mô Tả

Hệ thống AI đa nhiệm (Multi-task Learning) phân tích ảnh đáy mắt (fundus) từ bộ dữ liệu ODIR-5K:
- **Task 1**: Phân loại đa nhãn 8 bệnh lý mắt (N, D, G, C, A, H, M, O)
- **Task 2**: Dự đoán tuổi sinh học (Retinal Age)

So sánh 2 kiến trúc: **EfficientNet-B0 (CNN)** vs **Swin Transformer**

---

## Cấu Trúc Thư Mục

```
DOANTOTNGHIEP/
│
├── archive/                            # DỮ LIỆU
│   ├── ODIR-5K/                        #   Ảnh gốc (7,000 ảnh)
│   ├── preprocessed_images/            #   Ảnh ROI Crop 512×512 (6,392 ảnh)
│   ├── enhanced_images/                #   + Ben Graham + CLAHE (6,392 ảnh)
│   └── splits_clean/                   #   CSV splits + metadata.json
│       ├── train.csv                   #     4,462 ảnh (~70%)
│       ├── val.csv                     #     948 ảnh (~15%)
│       ├── test.csv                    #     954 ảnh (~15%)
│       └── metadata.json              #     pos_weight, age_mean/std
│
├── configs/                            # CẤU HÌNH 6 THỰC NGHIỆM
│   ├── exp_1_cnn_no_preprocess.yaml    #   EXP 1: CNN + ảnh gốc
│   ├── exp_2_cnn_preprocess_no_aug.yaml#   EXP 2: CNN + enhanced
│   ├── exp_3_cnn_preprocess_with_aug.yaml# EXP 3: CNN + enhanced + MixUp/CutMix
│   ├── exp_4_swin_no_preprocess.yaml   #   EXP 4: Swin + ảnh gốc
│   ├── exp_5_swin_preprocess_no_aug.yaml#  EXP 5: Swin + enhanced
│   └── exp_6_swin_preprocess_with_aug.yaml# EXP 6: Swin + enhanced + MixUp/CutMix
│
├── src/                                # MODULES TRAINING
│   ├── __init__.py                     #   Package exports
│   ├── dataset.py                      #   ODIRDataset, get_dataloaders
│   ├── transforms.py                   #   Albumentations (train/val, 224/384)
│   ├── utils.py                        #   pos_weight, age norm, metrics
│   ├── loss.py                         #   MultiTaskLoss (BCE + SmoothL1)
│   ├── mixup.py                        #   MixUpCollator (α=0.4)
│   ├── cutmix.py                       #   CutMixCollator (α=1.0)
│   └── models/
│       ├── __init__.py                 #   build_model(model_type='cnn'/'swin')
│       ├── efficientnet_mtl.py         #   EfficientNet-B0 Multi-task
│       └── swin_mtl.py                 #   Swin-Tiny Multi-task
│
├── scripts/                            # SCRIPTS TIỀN XỬ LÝ (chạy 1 lần)
│   ├── preprocess_enhance.py           #   ROI Crop → Ben Graham → CLAHE
│   ├── build_patient_splits.py         #   Patient-level split (chống leakage)
│   ├── check_preprocessing.py          #   8 bài test QA tự động
│   └── clean_and_rebuild.py            #   Loại tuổi bất thường, rebuild splits
│
├── tests/                              # UNIT TESTS
│   ├── test_mixup.py                   #   26 tests cho MixUpCollator
│   └── test_cutmix.py                  #   37 tests cho CutMixCollator
│
├── notebooks/                          # KAGGLE NOTEBOOKS
│   ├── odir5k_cnn_kaggle.ipynb         #   Notebook chạy EXP 1-2-3 (CNN)
│   └── kaggle_setup.md                 #   Hướng dẫn setup Kaggle từng bước
│
├── kaggle_upload/                      # UPLOAD CODE LÊN KAGGLE
│   └── upload_code.sh                  #   Script tự động upload
│
├── docs/                               # TÀI LIỆU BÁO CÁO
│   ├── De_cuong_DATN.pdf               #   Đề cương đồ án tốt nghiệp
│   ├── decuong_extracted.txt            #   Nội dung đề cương (text)
│   ├── Bao_cao_Tien_xu_ly_Du_lieu.md   #   Báo cáo tiền xử lý dữ liệu
│   ├── Cong_nghe_Mo_hinh_Du_an.md      #   Công nghệ và mô hình
│   ├── Giai_thich_Quy_trinh_Chi_tiet.md#   Giải thích pipeline chi tiết
│   ├── Giai_thich_Code_Tien_xu_ly.md   #   Giải thích code preprocessing
│   ├── Giai_thich_Dataset_ODIR.md      #   Giải thích dataset ODIR-5K
│   ├── Y_nghia_Tien_xu_ly_Anh_Y_te.md  #   Ý nghĩa tiền xử lý ảnh y tế
│   ├── Tac_dung_File_Python.md          #   Tác dụng các file Python
│   ├── Tien_Do_Tong_Hop.md              #   Tổng hợp tiến độ
│   ├── check_result_utf8.txt            #   Kết quả kiểm tra dữ liệu
│   └── README_split.md                  #   Giải thích chia tập dữ liệu
│
├── results/                            # KẾT QUẢ TRAINING (tạo tự động)
│
├── train.py                            # ENTRY POINT — huấn luyện mô hình
├── evaluate.py                         # SO SÁNH kết quả 6 thực nghiệm
├── run_experiment.sh                   # AUTOMATION — chạy tuần tự các EXP
├── .gitignore                          # Loại trừ data nặng, cache
└── README.md                           # File này
```

---

## Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.9+ |
| Framework DL | PyTorch, timm |
| Xử lý ảnh | OpenCV, Albumentations |
| Dữ liệu | pandas, numpy, scikit-learn |
| Huấn luyện | Kaggle Notebooks (GPU T4) |

---

## Ma Trận 6 Thực Nghiệm (Ablation Study)

|  | Ảnh gốc (Raw) | Enhanced (ROI+BG+CLAHE) | Enhanced + MixUp/CutMix |
|--|---------------|------------------------|------------------------|
| **EfficientNet-B0** | EXP 1 | EXP 2 | EXP 3 |
| **Swin-Tiny** | EXP 4 | EXP 5 | EXP 6 |

**So sánh:**
- Dọc (EXP 1→2→3 hoặc 4→5→6): Đóng góp của tiền xử lý và augmentation
- Ngang (EXP 3 vs 6): CNN vs Swin Transformer

---

## Cách Chạy

### Trên Kaggle (khuyến nghị):
```bash
# 1. Upload code lên Kaggle Dataset
bash kaggle_upload/upload_code.sh

# 2. Tạo notebook trên kaggle.com, gắn 2 datasets, bật GPU
# 3. Upload notebooks/odir5k_cnn_kaggle.ipynb → Run All
```

### Trên local (chỉ test pipeline):
```bash
source .venv/bin/activate
python train.py --config configs/exp_1_cnn_no_preprocess.yaml
```

Xem hướng dẫn chi tiết: [notebooks/kaggle_setup.md](notebooks/kaggle_setup.md)
