# BẢNG SO SÁNH ABLATION — ODIR-5K Phase 1 (Nhị phân Siamese song nhãn)

Chỉ số trên **tập Test** ở **ngưỡng tối ưu Youden** (tìm trên Validation). Nhãn: 0 = Normal, 1 = Pathological.

| EXP | Kiến trúc | Ảnh | Aug | Accuracy | AUC-ROC | F1 | Sensitivity | Specificity | Age MAE |
| :--: | :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| EXP 1 cnn raw | EfficientNet-B0 | raw (ảnh gốc) | ❌ | 0.6614 | 0.7546 | 0.7099 | 0.6118 | 0.7654 | 8.02y |
| EXP 2 cnn enhanced | EfficientNet-B0 | enhanced | ❌ | 0.6693 | 0.7911 | 0.6993 | 0.5676 | 0.8827 | 7.92y |
| EXP 3 cnn enhanced aug | EfficientNet-B0 | enhanced | ✅ | 0.6912 | 0.7840 | 0.7446 | 0.6647 | 0.7469 | 7.74y |
| EXP 4 swin raw | Swin-Tiny | raw (ảnh gốc) | ❌ | 0.7311 | 0.8200 | 0.7900 | 0.7471 | 0.6975 | 7.41y |
| EXP 5 swin enhanced | Swin-Tiny | enhanced | ❌ | 0.7629 | 0.8563 | 0.8046 | 0.7206 | 0.8519 | 7.96y |
| EXP 6 swin enhanced aug | Swin-Tiny | enhanced | ✅ | 0.7291 | 0.8012 | 0.7695 | 0.6676 | 0.8580 | 7.58y |
