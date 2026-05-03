"""
Test toàn diện MixUpCollator cho ODIR-5K.
Chạy: python scripts/test_mixup.py
"""
import sys
import traceback
sys.path.insert(0, ".")

import torch
import numpy as np
from src.mixup import MixUpCollator, get_mixup_dataloader

PASS = "\u2705 PASS"
FAIL = "\u274c FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_batch(n=8, img_size=224):
    """Tạo batch giả lập: ảnh fundus + nhãn 8 lớp + tuổi."""
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
print("KIEM TRA MIXUP COLLATOR — ODIR-5K")
print("=" * 60)

# --- 1. Khởi tạo ---
print("\n[1] Khoi tao MixUpCollator")
try:
    c = MixUpCollator(alpha=0.4, prob=0.5, seed=42)
    check("Khoi tao thanh cong voi tham so hop le", True)
except Exception as e:
    check("Khoi tao thanh cong voi tham so hop le", False, str(e))

try:
    MixUpCollator(alpha=-0.1)
    check("Bat loi alpha <= 0", False, "Khong raise ValueError")
except ValueError:
    check("Bat loi alpha <= 0", True)

try:
    MixUpCollator(alpha=0.4, prob=0.0)
    check("Bat loi prob = 0", False, "Khong raise ValueError")
except ValueError:
    check("Bat loi prob = 0", True)

try:
    MixUpCollator(alpha=0.4, prob=1.5)
    check("Bat loi prob > 1", False, "Khong raise ValueError")
except ValueError:
    check("Bat loi prob > 1", True)

# --- 2. Shape output ---
print("\n[2] Kiem tra shape output")
collator = MixUpCollator(alpha=0.4, prob=1.0, seed=42)  # prob=1.0 -> luon MixUp
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
      isinstance(out["filename"], list) and len(out["filename"]) == 8,
      f"type={type(out['filename'])}, len={len(out['filename'])}")
check("mixup_lambda la Tensor",
      isinstance(out["mixup_lambda"], torch.Tensor))
check("mixed = True khi prob=1.0",
      out["mixed"] == True)

# --- 3. Tính đúng đắn của MixUp ---
print("\n[3] Kiem tra tinh dung dan MixUp")

lam = out["mixup_lambda"].item()
check("lambda >= 0.5 (dam bao anh A chiem ty le lon hon)",
      lam >= 0.5,
      f"lam={lam:.4f}")
check("lambda trong [0.5, 1.0]",
      0.5 <= lam <= 1.0,
      f"lam={lam:.4f}")

# Nhãn phải là soft (có giá trị trong (0,1) khi trộn 2 nhãn khác nhau)
labels = out["labels"]
check("Gia tri pixel image trong [0,1]",
      (out["image"] >= 0).all() and (out["image"] <= 1).all())
check("Gia tri labels trong [0,1]",
      (labels >= 0.0).all() and (labels <= 1.0).all())
check("Ton tai soft label (gia tri (0,1)) khi tron nhan khac nhau",
      ((labels > 0) & (labels < 1)).any().item())

# Ảnh phải khác ảnh gốc (đã bị trộn)
orig_images = torch.stack([b["image"] for b in batch])
check("Anh da bi tron (khac anh goc)",
      not torch.allclose(orig_images, out["image"]))

# Tuổi phải nằm trong khoảng [min_age, max_age] của batch
ages_orig = torch.stack([b["age"] for b in batch])
check("Tuoi mixed nam trong khoang min-max cua batch",
      (out["age"] >= ages_orig.min()).all() and
      (out["age"] <= ages_orig.max()).all())

# --- 4. Không MixUp khi prob thấp ---
print("\n[4] Kiem tra prob=0.01 (hiem khi tron)")
c_low = MixUpCollator(alpha=0.4, prob=0.01, seed=0)
n_mixed = sum(c_low(make_batch(8))["mixed"] for _ in range(50))
check("Trong 50 batches, it hon 5 batch duoc tron",
      n_mixed < 5,
      f"n_mixed={n_mixed}/50")

# --- 5. Luôn MixUp khi prob=1.0 ---
print("\n[5] Kiem tra prob=1.0 (luon tron)")
c_full = MixUpCollator(alpha=0.4, prob=1.0, seed=7)
n_mixed2 = sum(c_full(make_batch(8))["mixed"] for _ in range(10))
check("Trong 10 batches, tat ca deu duoc tron",
      n_mixed2 == 10,
      f"n_mixed={n_mixed2}/10")

# --- 6. Seed reproducibility ---
print("\n[6] Kiem tra reproducibility (same seed)")
c1 = MixUpCollator(alpha=0.4, prob=1.0, seed=123)
c2 = MixUpCollator(alpha=0.4, prob=1.0, seed=123)
b1 = c1(make_batch(8))
b2 = c2(make_batch(8))
check("Cung seed tao ra cung mixup_lambda",
      torch.allclose(b1["mixup_lambda"], b2["mixup_lambda"]),
      f"lam1={b1['mixup_lambda'].item():.6f}, lam2={b2['mixup_lambda'].item():.6f}")

# --- 7. Nhiều batch size ---
print("\n[7] Kiem tra batch size khac nhau")
for bs in [2, 4, 16, 32]:
    try:
        c_bs = MixUpCollator(alpha=0.4, prob=1.0)
        out_bs = c_bs(make_batch(bs))
        ok = out_bs["image"].shape[0] == bs
        check(f"batch_size={bs}: shape[0]=={bs}", ok, str(out_bs["image"].shape))
    except Exception as e:
        check(f"batch_size={bs}: khong loi", False, str(e))

# --- 8. Multi-task: age mixed đúng ---
print("\n[8] Kiem tra multi-task age mixing")
c_age = MixUpCollator(alpha=0.4, prob=1.0, seed=42)
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
lam_a = out_age["mixup_lambda"].item()
mixed_ages = out_age["age"].squeeze().tolist()
check("Tuoi mixed la noi suy tuyen tinh cua [40, 80]",
      all(40.0 <= a <= 80.0 for a in mixed_ages),
      f"ages={[f'{a:.1f}' for a in mixed_ages]}, lam={lam_a:.3f}")

# --- 9. repr ---
print("\n[9] Kiem tra repr")
r = repr(MixUpCollator(alpha=0.3, prob=0.6))
check("repr chua alpha va prob",
      "alpha=0.3" in r and "prob=0.6" in r,
      r)

# ── Tổng kết ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print()
print("=" * 60)
print(f"KET QUA: {n_pass}/{n_pass+n_fail} tests PASS")
if n_fail == 0:
    print("✅ MixUp PASS TOAN BO — San sang cho buoc tiep theo")
else:
    print(f"❌ {n_fail} test FAIL — Can kiem tra lai")
    for name, ok, detail in results:
        if not ok:
            print(f"   FAIL: {name} [{detail}]")
print("=" * 60)
sys.exit(0 if n_fail == 0 else 1)
