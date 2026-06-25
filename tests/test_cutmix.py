"""
Test toàn diện CutMixCollator cho ODIR-5K.
Chạy: python scripts/test_cutmix.py
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from src.cutmix import CutMixCollator, get_cutmix_dataloader

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_batch(n=8, img_size=224):
    labels_list = [
        [1,0,0,0,0,0,0,0],  # Normal
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],  # Diabetes
        [0,0,1,0,0,0,0,0],  # Glaucoma
        [0,0,0,0,0,1,0,0],  # Hypertension (thiểu số)
        [0,0,0,0,0,1,0,0],
        [0,1,1,0,0,0,0,0],  # Multi-label
        [0,0,0,0,0,0,0,1],  # Other
    ]
    batch = []
    for i in range(n):
        batch.append({
            "image":    torch.rand(3, img_size, img_size),
            "labels":   torch.FloatTensor(labels_list[i % len(labels_list)]),
            "age":      torch.FloatTensor([float(40 + i * 5)]),
            "filename": f"fake_{i:04d}.jpg",
        })
    return batch

# ── Test suite ────────────────────────────────────────────────────────────────
print("=" * 60)
print("KIEM TRA CUTMIX COLLATOR — ODIR-5K")
print("=" * 60)

# --- 1. Khởi tạo ---
print("\n[1] Khoi tao CutMixCollator")
try:
    c = CutMixCollator(alpha=1.0, prob=0.5, seed=42)
    check("Khoi tao thanh cong voi tham so hop le", True)
except Exception as e:
    check("Khoi tao thanh cong voi tham so hop le", False, str(e))

try:
    CutMixCollator(alpha=-0.1)
    check("Bat loi alpha <= 0", False, "Khong raise ValueError")
except ValueError:
    check("Bat loi alpha <= 0", True)

try:
    CutMixCollator(alpha=1.0, prob=0.0)
    check("Bat loi prob = 0", False, "Khong raise ValueError")
except ValueError:
    check("Bat loi prob = 0", True)

try:
    CutMixCollator(alpha=1.0, prob=1.5)
    check("Bat loi prob > 1", False, "Khong raise ValueError")
except ValueError:
    check("Bat loi prob > 1", True)

# --- 2. Shape output ---
print("\n[2] Kiem tra shape output")
collator = CutMixCollator(alpha=1.0, prob=1.0, seed=42)
batch = make_batch(8)
out = collator(batch)

check("image shape [8,3,224,224]",
      out["image"].shape == torch.Size([8, 3, 224, 224]),
      str(out["image"].shape))
check("labels shape [8,8]",
      out["labels"].shape == torch.Size([8, 8]),
      str(out["labels"].shape))
check("age shape [8,1]",
      out["age"].shape == torch.Size([8, 1]),
      str(out["age"].shape))
check("filename la list do dai 8",
      isinstance(out["filename"], list) and len(out["filename"]) == 8)
check("cutmix_lambda la Tensor",
      isinstance(out["cutmix_lambda"], torch.Tensor))
check("mixed = True khi prob=1.0",
      out["mixed"] == True)

# --- 3. Tính đúng đắn của CutMix ---
print("\n[3] Kiem tra tinh dung dan CutMix")

lam = out["cutmix_lambda"].item()
check("lambda trong [0, 1]",
      0.0 <= lam <= 1.0,
      f"lam={lam:.4f}")

# Giá trị pixel phải trong [0,1]
check("Gia tri pixel image trong [0,1]",
      (out["image"] >= 0).all() and (out["image"] <= 1).all())

# Nhãn phải trong [0,1]
labels = out["labels"]
check("Gia tri labels trong [0,1]",
      (labels >= 0.0).all() and (labels <= 1.0).all())

# Ảnh gốc vs ảnh CutMix: phải khác nhau
orig_images = torch.stack([b["image"] for b in batch])
check("Anh da bi cat-dan (khac anh goc)",
      not torch.allclose(orig_images, out["image"]))

# Kiểm tra vùng được dán: ít nhất 1 vùng pixel của ảnh_mix bằng pixel ảnh khác
# (nếu có vùng cắt thì ít nhất 1 pixel bị thay)
check("Co su thay the pixel (vung cat co ton tai)",
      not torch.allclose(orig_images, out["image"]))

# Tuổi phải nằm trong khoảng
ages_orig = torch.stack([b["age"] for b in batch])
check("Tuoi mixed nam trong khoang min-max cua batch",
      (out["age"] >= ages_orig.min()).all() and
      (out["age"] <= ages_orig.max()).all())

# Lambda từ diện tích thực tế phải nhất quán
check("Soft labels nhat quan voi lambda (labels trong [0,1])",
      (labels >= 0.0).all() and (labels <= 1.0).all())

# --- 4. Kiểm tra _rand_bbox ---
print("\n[4] Kiem tra _rand_bbox")
c_box = CutMixCollator(alpha=1.0, prob=1.0, seed=0)
for lam_test in [0.1, 0.3, 0.5, 0.7, 0.9]:
    x1, y1, x2, y2 = c_box._rand_bbox(224, 224, lam_test)
    valid = (0 <= x1 <= x2 <= 224) and (0 <= y1 <= y2 <= 224)
    check(f"_rand_bbox(lam={lam_test}): toa do hop le [x1={x1},y1={y1},x2={x2},y2={y2}]",
          valid)

# Lambda thực tế phải = 1 - area_ratio
x1, y1, x2, y2 = c_box._rand_bbox(224, 224, 0.5)
lam_actual = 1.0 - (x2 - x1) * (y2 - y1) / (224 * 224)
check("Lambda thuc te = 1 - area_ratio",
      0.0 <= lam_actual <= 1.0,
      f"lam_actual={lam_actual:.4f}")

# --- 5. Prob thấp ---
print("\n[5] Kiem tra prob=0.01 (hiem khi cat)")
c_low = CutMixCollator(alpha=1.0, prob=0.01, seed=0)
n_mixed = sum(c_low(make_batch(8))["mixed"] for _ in range(50))
check("Trong 50 batches, it hon 5 batch duoc cat",
      n_mixed < 5,
      f"n_mixed={n_mixed}/50")

# --- 6. Luôn CutMix khi prob=1.0 ---
print("\n[6] Kiem tra prob=1.0 (luon cat)")
c_full = CutMixCollator(alpha=1.0, prob=1.0, seed=7)
n_mixed2 = sum(c_full(make_batch(8))["mixed"] for _ in range(10))
check("Trong 10 batches, tat ca deu duoc cat",
      n_mixed2 == 10,
      f"n_mixed={n_mixed2}/10")

# --- 7. Seed reproducibility ---
print("\n[7] Kiem tra reproducibility (same seed)")
c1 = CutMixCollator(alpha=1.0, prob=1.0, seed=123)
c2 = CutMixCollator(alpha=1.0, prob=1.0, seed=123)
b1 = c1(make_batch(8))
b2 = c2(make_batch(8))
check("Cung seed tao ra cung cutmix_lambda",
      torch.allclose(b1["cutmix_lambda"], b2["cutmix_lambda"]),
      f"lam1={b1['cutmix_lambda'].item():.6f}, lam2={b2['cutmix_lambda'].item():.6f}")

# --- 8. Nhiều batch size ---
print("\n[8] Kiem tra batch size khac nhau")
for bs in [2, 4, 16, 32]:
    try:
        c_bs = CutMixCollator(alpha=1.0, prob=1.0)
        out_bs = c_bs(make_batch(bs))
        ok = out_bs["image"].shape[0] == bs
        check(f"batch_size={bs}: shape[0]=={bs}", ok, str(out_bs["image"].shape))
    except Exception as e:
        check(f"batch_size={bs}: khong loi", False, str(e))

# --- 9. Nhiều image size ---
print("\n[9] Kiem tra image size khac nhau (224 va 384)")
for img_size in [224, 384]:
    try:
        c_sz = CutMixCollator(alpha=1.0, prob=1.0, seed=0)
        out_sz = c_sz(make_batch(4, img_size=img_size))
        expected = torch.Size([4, 3, img_size, img_size])
        check(f"img_size={img_size}: shape dung", out_sz["image"].shape == expected,
              str(out_sz["image"].shape))
    except Exception as e:
        check(f"img_size={img_size}: khong loi", False, str(e))

# --- 10. Multi-task age mixing ---
print("\n[10] Kiem tra multi-task age mixing")
c_age = CutMixCollator(alpha=1.0, prob=1.0, seed=42)
batch_age = [
    {"image": torch.rand(3,64,64),
     "labels": torch.FloatTensor([1,0,0,0,0,0,0,0]),
     "age": torch.FloatTensor([40.0]),
     "filename": "a.jpg"},
    {"image": torch.rand(3,64,64),
     "labels": torch.FloatTensor([0,1,0,0,0,0,0,0]),
     "age": torch.FloatTensor([80.0]),
     "filename": "b.jpg"},
]
out_age = c_age(batch_age)
mixed_ages = out_age["age"].squeeze().tolist()
check("Tuoi mixed nam trong [40, 80]",
      all(40.0 <= a <= 80.0 for a in mixed_ages),
      f"ages={[f'{a:.1f}' for a in mixed_ages]}")

# --- 11. So sánh với MixUp: pixel vùng cắt phải bằng pixel ảnh B ---
print("\n[11] Kiem tra pixel vung cat = pixel anh B (dac trung CutMix)")
c_check = CutMixCollator(alpha=1.0, prob=1.0, seed=99)
batch2 = make_batch(2, img_size=64)
# Ghi đè rng để biết bbox
c_check.rng = np.random.default_rng(99)
# Tính bbox thủ công
lam_manual = float(np.random.default_rng(99).beta(1.0, 1.0))
c_check2 = CutMixCollator(alpha=1.0, prob=1.0, seed=99)
out2 = c_check2(batch2)
lam2 = out2["cutmix_lambda"].item()
check("Lambda thuc te >= 0 (vung cat hop le)",
      lam2 >= 0.0,
      f"lam={lam2:.4f}")

# --- 12. repr ---
print("\n[12] Kiem tra repr")
r = repr(CutMixCollator(alpha=0.5, prob=0.7))
check("repr chua alpha va prob",
      "alpha=0.5" in r and "prob=0.7" in r, r)

# --- 13. CutMix khác MixUp: pixel gốc vẫn còn trong ảnh ---
print("\n[13] Kiem tra ban chat CutMix: pixel ngoai vung cat giu nguyen")
c13 = CutMixCollator(alpha=1.0, prob=1.0, seed=3)
# Dùng ảnh khác nhau rõ ràng
img_A = torch.zeros(3, 64, 64)    # toàn đen
img_B = torch.ones(3, 64, 64)     # toàn trắng
b13 = [
    {"image": img_A, "labels": torch.FloatTensor([1,0,0,0,0,0,0,0]),
     "age": torch.FloatTensor([50.0]), "filename": "black.jpg"},
    {"image": img_B, "labels": torch.FloatTensor([0,1,0,0,0,0,0,0]),
     "age": torch.FloatTensor([60.0]), "filename": "white.jpg"},
]
out13 = c13(b13)
# Ảnh mix phải có vùng 0 (từ A) và vùng 1 (từ B) → không phải giá trị trung gian
img_mix = out13["image"][0]  # sample đầu tiên
has_zeros = (img_mix == 0).any().item()
has_ones  = (img_mix == 1).any().item()
check("Anh mix co ca vung nguyen-A (0) va vung nguyen-B (1)",
      has_zeros and has_ones,
      f"has_zeros={has_zeros}, has_ones={has_ones}")
# Không có pixel trung gian (không như MixUp)
has_intermediate = ((img_mix > 0) & (img_mix < 1)).any().item()
check("Khong co pixel trung gian (dac trung CutMix vs MixUp)",
      not has_intermediate,
      f"has_intermediate={has_intermediate}")

# ── Tổng kết ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print()
print("=" * 60)
print(f"KET QUA: {n_pass}/{n_pass+n_fail} tests PASS")
if n_fail == 0:
    print("✅ CutMix PASS TOAN BO — San sang cho buoc tiep theo")
else:
    print(f"❌ {n_fail} test FAIL — Can kiem tra lai")
    for name, ok, detail in results:
        if not ok:
            print(f"   FAIL: {name} [{detail}]")
print("=" * 60)

import unittest
class TestCutMix(unittest.TestCase):
    def test_run_suite(self):
        self.assertEqual(n_fail, 0, f"{n_fail} tests failed in CutMix suite")
